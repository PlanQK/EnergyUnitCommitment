from collections import OrderedDict
import numpy as np


class IsingPypsaInterface:
    def __init__(self, network, snapshots):
        self.problem = {}
        self.snapshots = snapshots
        self._startIndex = OrderedDict()
        count = 0
        for i in range(len(network.buses)):
            gen = network.generators[
                network.generators.bus == network.buses.index[i]
            ]
            for name in gen.index:
                if network.generators.committable[name]:
                    continue
                self._startIndex[name] = count
                count += len(self.snapshots)
        self.slackStart = count
        self.slackIndex = count

    def addInteraction(self, *args):
        if len(args) < 2:
            raise ValueError(
                "An interaction needs at least one spin id and a weight."
            )
        for i in range(len(args) - 1):
            if not isinstance(args[i], int):
                raise ValueError("The spin id needs to be an integer")
        if not isinstance(args[-1], float):
            raise ValueError("The interaction needs to be a float")
        if args[-1] != 0:
            key = tuple(sorted(args[:-1]))
            # the minus is necessary because the solver has an additional -1 factor in the couplings
            self.problem[key] = self.problem.get(key, 0) - args[-1]

    def toVecIndex(self, generator, time=0):
        pos = self._startIndex.get(generator, None)
        if pos is None:
            return None
        return pos + time

    def addSlackVar(self):
        var = self.slackIndex
        self.slackIndex += 1
        return var

    def fromVecIndex(self, index):
        for key in self._startIndex:
            if index < self._startIndex[key] + len(self.snapshots):
                break
        if self.slackStart <= index:
            raise ValueError("No Generator assigned to this index")
        return (key, index % len(self.snapshots))

    def siquanFormat(self):
        return [(v, list(k)) for k, v in self.problem.items()]

    def numVariables(self):
        return self.slackIndex

    @classmethod
    def buildCostFunction(
        cls,
        network,
        monetaryCostFactor,
        kirchoffFactor,
        minUpDownFactor,
        useKirchoffInequality=True,
    ):
        problemDict = cls(network, network.snapshots)

        # TODO these still need to be adjusted
        lambdaMinUp = minUpDownFactor  # for Tminup constraint
        lambdaMinDown = minUpDownFactor  # for Tmindown constraint
        for gen in problemDict._startIndex:
            for t in range(len(problemDict.snapshots)):
                # operating cost contribution
                index = problemDict.toVecIndex(gen, t)
                val = network.generators["marginal_cost"].loc[gen] * (
                    network.generators_t["p"].iloc[t].loc[gen]
                )
                problemDict.addInteraction(index, monetaryCostFactor * val)

                # startup and shutdown costs
                su = network.generators["start_up_cost"].loc[gen]
                sd = network.generators["shut_down_cost"].loc[gen]

                Tmindown = min(
                    len(problemDict.snapshots) - t - 1,
                    network.generators["min_down_time"].loc[gen],
                )
                Tminup = min(
                    len(problemDict.snapshots) - t - 1,
                    network.generators["min_up_time"].loc[gen],
                )

                if t == 0:
                    problemDict.addInteraction(
                        index, 0.5 * monetaryCostFactor * (sd - su)
                    )

                elif t == len(problemDict.snapshots) - 1:
                    problemDict.addInteraction(
                        index, 0.5 * monetaryCostFactor * (su - sd)
                    )

                else:
                    problemDict.addInteraction(
                        index,
                        index - 1,
                        -0.5 * monetaryCostFactor * (su + sd)
                        - lambdaMinUp * Tminup
                        - lambdaMinDown * Tmindown,
                    )
                    problemDict.addInteraction(
                        index, lambdaMinUp * Tminup - lambdaMinDown * Tmindown
                    )
                    problemDict.addInteraction(
                        index - 1,
                        -lambdaMinUp * Tminup + lambdaMinDown * Tmindown,
                    )
                    for deltaT in range(Tminup):
                        problemDict.addInteraction(
                            index + deltaT + 1, -lambdaMinUp
                        )
                        problemDict.addInteraction(
                            index, index + deltaT + 1, -lambdaMinUp
                        )
                        problemDict.addInteraction(
                            index - 1, index + deltaT + 1, lambdaMinUp
                        )
                        problemDict.addInteraction(
                            index, index - 1, index + deltaT + 1, lambdaMinUp
                        )

                    for deltaT in range(Tmindown):
                        problemDict.addInteraction(
                            index + deltaT + 1, lambdaMinDown
                        )
                        problemDict.addInteraction(
                            index - 1, index + deltaT + 1, lambdaMinDown
                        )
                        problemDict.addInteraction(
                            index, index + deltaT + 1, -lambdaMinDown
                        )
                        problemDict.addInteraction(
                            index,
                            index - 1,
                            index + deltaT + 1,
                            -lambdaMinDown,
                        )

        if useKirchoffInequality:
            IsingPypsaInterface._kirchoffInequalityConstraint(
                network, problemDict, kirchoffFactor
            )
        else:
            IsingPypsaInterface._kirchhoffEqualityConstraint(
                network, problemDict, kirchoffFactor
            )
        return problemDict

    @staticmethod
    def getGeneratorOutput(network, gen, time):
        factor = 1.0
        if gen in network.generators_t.p_max_pu:
            factor = network.generators_t.p_max_pu[gen].iloc[time]
        # return np.round(factor*network.generators.p_nom[gen])
        return factor * network.generators.p_nom[gen]

    @staticmethod
    def generateConstant(network, node, t):
        # the helper c_nt calculation
        c_nt = 0.0
        # storage
        for i in network.storage_units[
            network.storage_units.bus == node
        ].index:
            c_nt += network.storage_units_t.p[i].iloc[t]
        # lines
        for i in network.lines[network.lines.bus0 == node].index:
            c_nt -= network.lines_t.p0[i].iloc[t]
        for i in network.lines[network.lines.bus1 == node].index:
            c_nt -= network.lines_t.p1[i].iloc[t]
        # loads contribution
        for load in network.loads[network.loads.bus == node].index:
            c_nt -= network.loads_t.p_set[load].iloc[t]
        return c_nt

    @staticmethod
    def _kirchoffEqualityConstraint(network, problemDict, lambda2):
        """
        This constraint implements the equality as a constraint.
        It requires less overhead but forces the solution to a trivial 1 state.
        """
        # kirchhoff constraint (sums over s)
        for node in network.buses.index:
            for t in range(len(problemDict.snapshots)):
                c_nt = IsingPypsaInterface.generateConstant(network, node, t)
                # the kirchoff contributions start here
                for gen0 in network.generators[
                    network.generators.bus == node
                ].index:
                    index = problemDict.toVecIndex(gen0, t)
                    if index is None:
                        continue
                    # quadratic sum
                    for gen1 in network.generators[
                        network.generators.bus == node
                    ].index:
                        index1 = problemDict.toVecIndex(gen1, t)
                        if index1 is None:
                            # val = getGeneratorOutput(network, gen0, t) * getGeneratorOutput(network, gen1, t)
                            val = (
                                network.generators_t.p.iloc[t].loc[gen0]
                                * network.generators_t.p.iloc[t].loc[gen1]
                            )
                            problemDict.addInteraction(
                                index, 0.5 * lambda2 * val
                            )
                        elif index == index1:
                            continue
                        else:
                            # val = getGeneratorOutput(network, gen0, t) * getGeneratorOutput(network, gen1, t)
                            val = (
                                network.generators_t.p.iloc[t].loc[gen0]
                                * network.generators_t.p.iloc[t].loc[gen1]
                            )
                            problemDict.addInteraction(
                                index, index1, 1.0 / 4 * lambda2 * val
                            )
                    # c_nt sum
                    nodePower = 0
                    for gen1 in network.generators[
                        network.generators.bus == node
                    ].index:
                        nodePower += network.generators_t.p.iloc[t].loc[gen1]
                    val = (
                        c_nt + 0.5 * nodePower
                    ) * network.generators_t.p.iloc[t].loc[gen0]
                    problemDict.addInteraction(index, lambda2 * val)

    @staticmethod
    def _kirchoffInequalityConstraint(network, problemDict, lambda2):
        """This implements a less or equal version of the kirchoff constraint using slack variables.
        It allows the optimization to end in configurations that have more energy than required. The
        fine-tuning is left for the powerflow optimization.
        """
        for node in network.buses.index:
            for t in range(len(problemDict.snapshots)):
                maxPossible = 0
                for gen in network.generators[
                    network.generators.bus == node
                ].index:
                    maxPossible += IsingPypsaInterface.getGeneratorOutput(
                        network, gen, t
                    )
                lenbinary = len("{0:b}".format(int(np.round(maxPossible))))
                totalBinarySum = 2 ** lenbinary - 1
                c_nt = IsingPypsaInterface.generateConstant(network, node, t)
                curSlack = problemDict.slackIndex

                for i in range(lenbinary):
                    problemDict.addSlackVar()
                    for j in range(lenbinary):
                        if j == i:
                            continue
                        problemDict.addInteraction(
                            curSlack + i,
                            curSlack + j,
                            lambda2 * (1.0 / 4 * 2 ** i * 2 ** j),
                        )
                    problemDict.addInteraction(
                        curSlack + i,
                        lambda2
                        * (0.5 * totalBinarySum - c_nt - 0.5 * maxPossible)
                        * 2 ** i,
                    )
                for gen0 in network.generators[
                    network.generators.bus == node
                ].index:
                    index = problemDict.toVecIndex(gen0, t)
                    if index is None:
                        continue
                    # quadratic sum
                    for gen1 in network.generators[
                        network.generators.bus == node
                    ].index:
                        index1 = problemDict.toVecIndex(gen1, t)
                        if index1 is None:
                            val = IsingPypsaInterface.getGeneratorOutput(
                                network, gen0, t
                            ) * IsingPypsaInterface.getGeneratorOutput(
                                network, gen1, t
                            )
                            problemDict.addInteraction(
                                index, 0.5 * lambda2 * val
                            )
                        elif index == index1:
                            continue
                        else:
                            val = IsingPypsaInterface.getGeneratorOutput(
                                network, gen0, t
                            ) * IsingPypsaInterface.getGeneratorOutput(
                                network, gen1, t
                            )
                            problemDict.addInteraction(
                                index, index1, 1.0 / 4 * lambda2 * val
                            )
                    # c_nt sum + total generator sum
                    val = (
                        c_nt + 0.5 * maxPossible - 0.5 * totalBinarySum
                    ) * IsingPypsaInterface.getGeneratorOutput(
                        network, gen0, t
                    )
                    problemDict.addInteraction(index, lambda2 * val)
                    # mixed slack+generator terms
                    for i in range(lenbinary):
                        val = (
                            -0.5
                            * 2 ** i
                            * IsingPypsaInterface.getGeneratorOutput(
                                network, gen0, t
                            )
                        )
                        problemDict.addInteraction(
                            index, curSlack + i, lambda2 * val
                        )

    @staticmethod
    def addSQASolutionToNetwork(network, problemDict, solutionState):
        for gen in problemDict._startIndex:
            vec = np.zeros(len(problemDict.snapshots))
            network.generators_t.status[gen] = np.concatenate(
                [
                    vec,
                    np.ones(
                        len(network.snapshots) - len(problemDict.snapshots)
                    ),
                ]
            )
        vec = np.zeros(len(problemDict.snapshots))
        gen, time = problemDict.fromVecIndex(0)
        for index in solutionState:
            try:
                new_gen, new_time = problemDict.fromVecIndex(index)
            except:
                continue
            if gen != new_gen:
                network.generators_t.status[gen] = np.concatenate(
                    [
                        vec,
                        np.ones(
                            len(network.snapshots) - len(problemDict.snapshots)
                        ),
                    ]
                )
                vec = np.zeros(len(problemDict.snapshots))
                gen = new_gen
                vec[new_time] = 1
        network.generators_t.status[gen] = np.concatenate(
            [vec, np.ones(len(network.snapshots) - len(problemDict.snapshots))]
        )
        return network
