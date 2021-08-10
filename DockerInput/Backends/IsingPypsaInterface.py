from collections import OrderedDict
import numpy as np
from EnvironmentVariableManager import EnvironmentVariableManager
import typing


class IsingPypsaInterface:
    def __init__(self, network, snapshots):
        self.problem = {}
        self.snapshots = snapshots
        self._startIndex = {}
        self.network = network

        envMgr = EnvironmentVariableManager()
        self.kirchhoffFactor = float(envMgr["kirchhoffFactor"])
        self.monetaryCostFactor = float(envMgr["monetaryCostFactor"])
        self.minUpDownFactor = float(envMgr["minUpDownFactor"])
        self.slackVarFactor = float(envMgr["slackVarFactor"])
        count = 0
        for i in range(len(self.network.buses)):
            gen = self.network.generators[
                self.network.generators.bus == self.network.buses.index[i]
            ]
            for name in gen.index:
                if self.network.generators.committable[name]:
                    continue
                self._startIndex[name] = count
                count += len(self.snapshots)

        # this allows to optimize the flow through lines
        # requires a binary representation of the number
        # and an additional variable for the sign
        self._lineIndices = {}
        self._lineDirection = {}

        for index in self.network.lines.index:
            # store the directional qubits first, then the line's binary representations
            self._lineDirection[index] = count
            self._lineIndices[index] = count + len(self.snapshots)
            count += len(self.snapshots) * (self.getBinaryLength(index) + 1)

        # to avoid cubic or quartic contributions we add slack variables that
        # store the product of the binary bit and the sign
        self._slackVarsForQubo = {}
        self._slackStart = count
        for index in self.network.lines.index:
            self._slackVarsForQubo[index] = count
            count += len(self.snapshots) * self.getBinaryLength(index)
        self.slackIndex = count

    def addInteraction(self, *args):
        """Helper function to define an Ising Interaction.

        Can take arbitrary number of arguments:
        The last argument is the interaction strength.
        The previous arguments contain the spin ids.
        """
        if len(args) < 2:
            raise ValueError(
                "An interaction needs at least one spin id and a weight."
            )
        if len(args) == 3 and args[0] == args[1]:
            raise ValueError("Same qubit")
        for i in range(len(args) - 1):
            if not isinstance(args[i], int):
                raise ValueError(
                    f"The spin id: {args[:-1]} needs to be an integer"
                )
        if not isinstance(args[-1], float):
            raise ValueError("The interaction needs to be a float")
        if args[-1] != 0:
            key = tuple(sorted(args[:-1]))
            # the minus is necessary because the solver has an additional
            # -1 factor in the couplings
            self.problem[key] = self.problem.get(key, 0) - args[-1]

    def toVecIndex(self, generator, time=0):
        """Return the index for the supplied generator at t=0.
        If the generator was not yet encountered, it creates a new
        set of indices for the generator and each point in time.
        """
        pos = self._startIndex.get(generator, None)
        if pos is None:
            return None
        return pos + time

    def getBinaryLength(self, line):
        return len(
            "{0:b}".format(int(np.round(self.network.lines.loc[line].s_nom)))
        )

    def lineToIndex(self, lineId, time=0):
        return self._lineIndices[lineId] + time * self.getBinaryLength(lineId)

    def lineDirectionToIndex(self, lineId, time=0):
        return self._lineDirection[lineId] + time * self.getBinaryLength(
            lineId
        )

    def lineSlackToIndex(self, lineId, time=0):
        return self._slackVarsForQubo[lineId] + time * self.getBinaryLength(
            lineId
        )

    def addSlackVar(self):
        """Return the index of a newly created slack variable.

        IMPORTANT: This function can only be used after all
        generators have been added and processed.
        This should only happen in the init function.
        """
        var = self.slackIndex
        self.slackIndex += 1
        return var

    def fromVecIndex(self, index):
        """Return a tuple of generator key and time from a given Spin ID."""
        for key in self._startIndex:
            if index < self._startIndex[key] + len(self.snapshots):
                break
        if self._slackStart <= index:
            raise ValueError("No Generator assigned to this index")
        return (key, index % len(self.snapshots))

    def siquanFormat(self):
        """Return the complete problem in the format for the siquan solver"""
        return [(v, list(k)) for k, v in self.problem.items()]

    def numVariables(self):
        """Return the number of spins."""
        return self.slackIndex

    def printIndividualCostContrib(self, result):
        for node in self.network.buses.index:
            for t in range(len(self.snapshots)):
                testNetwork = IsingPypsaInterface(self.network, self.snapshots)
                testNetwork._kirchhoffConstraint(node, t)
                print(f"\tCost Contrib for {node}, {t}")
                print(f"\t\t{testNetwork.problem}")
                print(
                    f"\t\t{testNetwork.calcCost(result)}, {testNetwork.constantCostContribution()}"
                )
        return

    def calcCost(self, result):
        result = set(result)
        totalCost = 0.0
        for spins, weight in self.problem.items():
            if len(spins) == 1:
                factor = 1
            else:
                factor = -1
            for spin in spins:
                if spin in result:
                    factor *= -1
            totalCost += factor * weight
        totalCost += self.constantCostContribution()
        return totalCost

    @classmethod
    def buildCostFunction(
        cls,
        network,
    ):
        """Build the complete cost function for an Ising formulation.
        The cost function is quite complex and I recommend first reading
        through the mathematical formulation.
        """
        problemDict = cls(network, network.snapshots)
        problemDict._marginalCosts()

        for gen in problemDict._startIndex:
            for t in range(len(problemDict.snapshots)):
                problemDict._startupShutdownCost(gen, t)

        # kirchhoff constraints
        for node in problemDict.network.buses.index:
            for t in range(len(problemDict.snapshots)):
                problemDict._kirchhoffConstraint(node, t)

        problemDict._addSlackConstraints()
        return problemDict

    def _marginalCosts(self):
        for index in self.network.lines.index:
            for t in range(len(self.snapshots)):
                lenbinary = self.getBinaryLength(index)
                for i in range(lenbinary):
                    self.addInteraction(
                        self._lineIndices[index] + i + lenbinary * t,
                        0.5 * self.monetaryCostFactor * 2 ** i,
                    )

        for gen in self._startIndex:
            for t in range(len(self.snapshots)):
                # operating cost contribution
                index = self.toVecIndex(gen, t)
                val = self.network.generators["marginal_cost"].loc[gen] * (
                    self.getGeneratorOutput(gen, t)
                )
                self.addInteraction(index, self.monetaryCostFactor * val)

    def _startupShutdownCost(self, gen, t):
        lambdaMinUp = self.minUpDownFactor  # for Tminup constraint
        lambdaMinDown = self.minUpDownFactor  # for Tmindown constraint
        index = self.toVecIndex(gen, t)
        su = self.network.generators["start_up_cost"].loc[gen]
        sd = self.network.generators["shut_down_cost"].loc[gen]

        Tmindown = min(
            len(self.snapshots) - t - 1,
            self.network.generators["min_down_time"].loc[gen],
        )
        Tminup = min(
            len(self.snapshots) - t - 1,
            self.network.generators["min_up_time"].loc[gen],
        )

        if t == 0:
            self.addInteraction(
                index, 0.5 * self.monetaryCostFactor * (sd - su)
            )

        elif t == len(self.snapshots) - 1:
            self.addInteraction(
                index, 0.5 * self.monetaryCostFactor * (su - sd)
            )

        else:
            self.addInteraction(
                index,
                index - 1,
                -0.5 * self.monetaryCostFactor * (su + sd)
                - lambdaMinUp * Tminup
                - lambdaMinDown * Tmindown,
            )
            self.addInteraction(
                index, lambdaMinUp * Tminup - lambdaMinDown * Tmindown
            )
            self.addInteraction(
                index - 1,
                -lambdaMinUp * Tminup + lambdaMinDown * Tmindown,
            )
            for deltaT in range(Tminup):
                self.addInteraction(index + deltaT + 1, -lambdaMinUp)
                self.addInteraction(index, index + deltaT + 1, -lambdaMinUp)
                self.addInteraction(index - 1, index + deltaT + 1, lambdaMinUp)
                self.addInteraction(
                    index, index - 1, index + deltaT + 1, lambdaMinUp
                )

            for deltaT in range(Tmindown):
                self.addInteraction(index + deltaT + 1, lambdaMinDown)
                self.addInteraction(
                    index - 1, index + deltaT + 1, lambdaMinDown
                )
                self.addInteraction(index, index + deltaT + 1, -lambdaMinDown)
                self.addInteraction(
                    index,
                    index - 1,
                    index + deltaT + 1,
                    -lambdaMinDown,
                )

    def getGeneratorOutput(self, gen, time):
        """Return the generator output for a given point in time.

        Two values can be used from the Pypsa datamodel:
            - p: The output of the LOPF optimization. This is not good, because
                after LOPF we will already have an optimized solution.
            - p_nom*p_max_pu: this does not need another optimization. The following
                parameters must be set:
                p_nom: the maximal output
                p_max_pu: a time dependent factor how much power can be output
                    (usually only set for renewables)
        """
        factor = 1.0
        if gen in self.network.generators_t.p_max_pu:
            factor = self.network.generators_t.p_max_pu[gen].iloc[time]
        return factor * self.network.generators.p_nom[gen]

    def getMaxLineEnergy(self, line):
        """Return the maximum value of energy a line can transport.

        Currently we round this to the next highest power of 2
        """
        return 2 ** self.getBinaryLength(line) - 1

    def maxGenPossible(self, node, t):
        maxPossible = 0
        for gen in self.network.generators[
            self.network.generators.bus == node
        ].index:
            maxPossible += self.getGeneratorOutput(gen, t)
        return maxPossible

    def totalLineCapacity(self, node):
        totalLineCapacity = 0.0
        lineIdsFactor = self.getLineFactors(node)
        for lineId1, factor in lineIdsFactor.items():
            totalLineCapacity += factor * self.getMaxLineEnergy(lineId1)
        return totalLineCapacity

    def generateConstant(self, node, t):
        # the helper c_nt calculation
        c_nt = 0.0
        # storage
        for i in self.network.storage_units[
            self.network.storage_units.bus == node
        ].index:
            c_nt += self.network.storage_units_t.p[i].iloc[t]
        # loads contribution
        for load in self.network.loads[self.network.loads.bus == node].index:
            c_nt -= self.network.loads_t.p_set[load].iloc[t]
        return c_nt

    def getLineFactors(self, node) -> typing.Dict[str, int]:
        lineIdsFactor = {}
        for lineId in self.network.lines[
            self.network.lines.bus0 == node
        ].index:
            lineIdsFactor[lineId] = 1
        for lineId in self.network.lines[
            self.network.lines.bus1 == node
        ].index:
            lineIdsFactor[lineId] = -1
        return lineIdsFactor

    def _quadraticGeneratorTerm(self, node, t):
        """This term is the quadratic contribution for the Kirchhoff constraint."""
        for gen0 in self.network.generators[
            self.network.generators.bus == node
        ].index:
            index = self.toVecIndex(gen0, t)
            if index is None:
                continue
            # generator^2 term
            for gen1 in self.network.generators[
                self.network.generators.bus == node
            ].index:
                index1 = self.toVecIndex(gen1, t)
                if index1 is None:
                    # this branch is for non-committable generators
                    # in our problems these should not appear
                    val = self.getGeneratorOutput(
                        gen0, t
                    ) * self.getGeneratorOutput(gen1, t)
                    self.addInteraction(
                        index, 0.5 * self.kirchhoffFactor * val
                    )
                elif index == index1:
                    # a squared variable does not affect the optimization problem
                    # we can therefore ignore it
                    continue
                else:
                    val = self.getGeneratorOutput(
                        gen0, t
                    ) * self.getGeneratorOutput(gen1, t)
                    self.addInteraction(
                        index, index1, 1.0 / 4 * self.kirchhoffFactor * val
                    )

    def _quadraticLineTerms(self, node, t):
        """This term is the quadratic contribution of the slack variables (lines)
        for the kirchhoff constraint.
        """
        lineIdsFactor = self.getLineFactors(node)
        for lineId1, factor1 in lineIdsFactor.items():
            lenbinary = self.getBinaryLength(lineId1)
            for i in range(lenbinary):
                for lineId2, factor2 in lineIdsFactor.items():
                    lenbinary2 = self.getBinaryLength(lineId2)
                    for j in range(lenbinary2):
                        # add the crossterm between line slack vars and line vars
                        # because the same loops are needed
                        self.addInteraction(
                            self.lineToIndex(lineId1, t) + i,
                            self.lineSlackToIndex(lineId2, t) + j,
                            -self.kirchhoffFactor
                            * 2 ** i
                            * 2 ** j
                            * factor1
                            * factor2,
                        )
                        if lineId1 != lineId2:
                            # cross term between total line sum and individual line
                            self.addInteraction(
                                self.lineToIndex(lineId1, t) + i,
                                self.lineToIndex(lineId2, t) + j,
                                self.kirchhoffFactor
                                / 4
                                * 2 ** i
                                * 2 ** j
                                * factor1
                                * factor2,
                            )
                            # slack vars make up the second quadratic sum
                            self.addInteraction(
                                self.lineSlackToIndex(lineId1, t) + i,
                                self.lineSlackToIndex(lineId2, t) + j,
                                self.kirchhoffFactor
                                * 2 ** i
                                * 2 ** j
                                * factor1
                                * factor2,
                            )

    def _linearGeneratorTerm(self, node, t):
        maxGenPossible = self.maxGenPossible(node, t)
        maxLineCapacity = self.totalLineCapacity(node)
        c_nt = self.generateConstant(node, t)
        for gen0 in self.network.generators[
            self.network.generators.bus == node
        ].index:
            index = self.toVecIndex(gen0, t)
            if index is None:
                continue
            self.addInteraction(
                index,
                self.kirchhoffFactor
                * (0.5 * maxGenPossible + 0.5 * maxLineCapacity + c_nt)
                * self.getGeneratorOutput(gen0, t),
            )

    def _linearLineTerms(self, node, t):
        maxGenPossible = self.maxGenPossible(node, t)
        maxLineCapacity = self.totalLineCapacity(node)
        c_nt = self.generateConstant(node, t)
        lineIdsFactor = self.getLineFactors(node)
        for lineId1, factor1 in lineIdsFactor.items():
            lenbinary = self.getBinaryLength(lineId1)
            for i in range(lenbinary):
                self.addInteraction(
                    self.lineSlackToIndex(lineId1, t) + i,
                    self.kirchhoffFactor
                    * (maxGenPossible + maxLineCapacity + 2 * c_nt)
                    * factor1
                    * 2 ** i,
                )
                self.addInteraction(
                    self.lineToIndex(lineId1, t) + i,
                    self.kirchhoffFactor
                    * (-0.5 * maxLineCapacity - c_nt - 0.5 * maxGenPossible)
                    * factor1
                    * 2 ** i,
                )

    def _crossTermGeneratorLine(self, node, t):
        lineIdsFactor = self.getLineFactors(node)
        for gen0 in self.network.generators[
            self.network.generators.bus == node
        ].index:
            index = self.toVecIndex(gen0, t)
            for lineId1, factor1 in lineIdsFactor.items():
                lenbinary = self.getBinaryLength(lineId1)
                for i in range(lenbinary):
                    val = (
                        self.kirchhoffFactor
                        * self.getGeneratorOutput(gen0, t)
                        * factor1
                        * 2 ** i
                    )
                    if index is None:
                        self.addInteraction(
                            self.lineToIndex(lineId1, t) + i,
                            -0.5 * val,
                        )
                        self.addInteraction(
                            self.lineSlackToIndex(lineId1, t) + i,
                            val,
                        )
                    else:
                        self.addInteraction(
                            index,
                            self.lineToIndex(lineId1, t) + i,
                            -0.5 * val,
                        )
                        self.addInteraction(
                            index,
                            self.lineSlackToIndex(lineId1, t) + i,
                            val,
                        )

    def _addSlackConstraints(self):
        for index in self.network.lines.index:
            lenbinary = self.getBinaryLength(index)
            for t in range(len(self.network.snapshots)):
                for i in range(lenbinary):
                    self.addInteraction(
                        self.lineToIndex(index, t) + i,
                        -self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineDirectionToIndex(index, t),
                        -self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineSlackToIndex(index, t) + i,
                        2 * self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineToIndex(index, t) + i,
                        self.lineDirectionToIndex(index, t),
                        self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineToIndex(index, t) + i,
                        self.lineSlackToIndex(index, t) + i,
                        -2 * self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineDirectionToIndex(index, t),
                        self.lineSlackToIndex(index, t) + i,
                        -2 * self.slackVarFactor,
                    )

    def constantCostContribution(self):
        totalConstCost = 0
        for node in self.network.buses.index:
            for t in range(len(self.snapshots)):
                maxGenPossible = self.maxGenPossible(node, t)
                maxLineCapacity = self.totalLineCapacity(node)
                c_nt = self.generateConstant(node, t)
                totalConstCost += self.kirchhoffFactor * (
                    0.25 * maxLineCapacity ** 2
                    + maxLineCapacity * c_nt
                    + 0.25 * maxGenPossible ** 2
                    + 0.5 * maxGenPossible * maxLineCapacity
                    + maxGenPossible * c_nt
                    + c_nt ** 2
                    # parts are from the quadratic sum where the gen, line, slacks are squared
                    + 0.25 * maxGenPossible ** 2
                    + 1.25 * maxLineCapacity ** 2
                )
        for index in self.network.lines.index:
            for t in range(len(self.snapshots)):
                lenbinary = self.getBinaryLength(index)
                totalConstCost += 3 * self.slackVarFactor * lenbinary
        return totalConstCost

    def _kirchhoffConstraint(self, node, t):
        self._quadraticGeneratorTerm(node, t)
        self._quadraticLineTerms(node, t)
        self._linearGeneratorTerm(node, t)
        self._linearLineTerms(node, t)
        self._crossTermGeneratorLine(node, t)

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
