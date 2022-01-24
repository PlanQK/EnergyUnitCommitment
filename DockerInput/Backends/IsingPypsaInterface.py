from collections import OrderedDict
import numpy as np
from EnvironmentVariableManager import EnvironmentVariableManager
import typing


class IsingPypsaInterface:

    # TODO Missing features for complete problem:
    # - marginal cost
    # - startup/shutdown cost
    # - multiple time slices

    def __init__(self, network, snapshots):

        # contains ising coefficients
        self.problem = {}

        self.network = network
        self.snapshots = snapshots

        self.allocatedQubits = 0
        # contains encoding data
        self.data = {}

        envMgr = EnvironmentVariableManager()
        self.kirchhoffFactor = float(envMgr["kirchhoffFactor"])
        self.monetaryCostFactor = float(envMgr["monetaryCostFactor"])
        self.minUpDownFactor = float(envMgr["minUpDownFactor"])
        self.slackVarFactor = float(envMgr["slackVarFactor"])

        # read generators and lines from network and encode as qubits
        self.storeGenerators()
        self.storeLines()


    def writeToHighestLevel(self, component):
        "write weights of qubits in highest level of data dictionary"
        for idx in range(len(self.data[component]['indices'])):
            self.data[self.data[component]['indices'][idx]] = self.data[component]['weights'][idx]


    def storeGenerators(self):
        "encodes generators at each time slice as a single qubit"
        for bus in range(len(self.network.buses)):
            generatorsAtBus = self.network.generators[
                self.network.generators.bus == self.network.buses.index[bus]
            ]
            for gen in generatorsAtBus.index:
                for time in range(len(self.network.snapshots)):
                    # no generator is supposed to be committable in our problems
                    if self.network.generators.committable[gen]:
                        continue
                    self.data[gen] = {
                            'indices' : [self.allocatedQubits],
                            'weights' : [self.network.generators_t.p_max_pu[gen].iloc[time]],
                            'encodingLength' : 1,
                    }
                    self.allocatedQubits += 1
                self.writeToHighestLevel(gen)


        return


    def storeLines(self):
        """
        wrapper for calling encodeLine to store a qubit representation
        of all lines at each time slice
        """
        for line in self.network.lines.index:
            for time in range(len(self.network.snapshots)):
            # overwrite this in child classes
                self.encodeLine(line,time)
            self.writeToHighestLevel(line)

    def encodeLine(self, line, time):
        """
        encodes a line at each time slice. splitCapacity gives weights of
        components that the line is split up into
        """
        capacity = int(self.network.lines.loc[line].s_nom)
        weights = self.splitCapacity(capacity)
        indices = list(range(self.allocatedQubits, self.allocatedQubits + len(weights),1))
        self.allocatedQubits += len(indices)
        self.data[line] = {
                'weights' : weights,
                'indices' : indices,
                'encodingLength' : len(weights),
        }
        return

    def splitCapacity(self, capacity):
        """returns a list of integers each representing a weight for a qubit.
        A collection of qubits with such a weight distributions represent a line
        with maximum power flow of capacity
        """
        return [1 for _ in range(0,capacity,1)]  + [-1 for _ in range(0,capacity,1)]

    # helper functions to obtain represented values

    def getBusComponents(self, bus):
        """return all labels of components that connect to a bus as a dictionary
        generators - at this bus
        loads - at this bus
        positiveLines - start in this bus
        negativeLines - end in this bus
        """
        result = {
                "generators":
                        list(self.network.generators[
                                self.network.generators.bus == bus
                        ].index)
                    ,
                "positiveLines" :
                        list(self.network.lines[
                                self.network.lines.bus0 == bus
                        ].index)
                ,
                "negativeLines" :
                        list(self.network.lines[
                                self.network.lines.bus1 == bus
                        ].index)
                ,
                }
        return result


    def getGeneratorStatus(self, gen, solution, time=0):
        return self.data[self.data[gen]['indices'][time]]

    def getLineFlow(self, line, solution, time=0):
        """
        returns the power flow through a line for qubits assignment according to solution"""
        result = 0
        for idx in range(len(self.data[line]['indices'])):
            if self.data[line]['indices'][idx] in solution:
                result+=self.data[line]['weights'][idx]
        return result


    def getFlowDictionary(self, solution):
        """build dictionary of all flows at each time slice"""
        solution = set(solution)
        result = {}
        for lineId in self.network.lines.index:
            for time in range(len(self.snapshots)):
                result[(lineId, time)] = self.getLineFlow(lineId, solution, time)
        return result


    def getLineValues(self, solution):
        return self.getFlowDictionary(solution)


    def getLoad(self, bus, time=0):
        loadsAtCurrentBus = self.network.loads[
                                    self.network.loads.bus == bus
                            ].index
        allLoads = self.network.loads_t['p_set'].iloc[time]
        result = allLoads[allLoads.index.isin(loadsAtCurrentBus)].sum()
        if result < 0:
            raise ValueError(
                "negative Load at current Bus"
            )
        return result


    def getMemoryAdress(self, component, time=0):
        encodingLength = self.data[component]["encodingLength"]
        return self.data[component]["indices"][time * encodingLength : (time+1) * encodingLength]


    # functions to couple components
    def addInteraction(self, *args):
        """Helper function to define an Ising Interaction.

        Can take arbitrary number of arguments:
        The last argument is the interaction strength.
        The previous arguments contain the spin ids.
        """
        if len(args) > 3:
            raise ValueError(
                "Too many arguments for an interaction"
            )

        key = tuple(args[:-1])
        interactionStrength = args[-1]
        for qubit in key:
            interactionStrength *= self.data[qubit]

        # if we couple two spins, we check if they are different. If both spins are the same, 
        # we substitute the product of spins with 1, since 1 * 1 = -1 * -1 = 1 holds. This
        # makes it into a constant contribution
        if len(key) == 2:
            if key[0] == key[1]:
                key = tuple([])
        self.problem[key] = self.problem.get(key,0) - interactionStrength


    def coupleComponentWithConstant(self, component, couplingStrength=1, time=0):
        """given a label, calculates the product with a fixed value and translates
        it into an ising interaction"""
        componentAdress = self.getMemoryAdress(component,time)
        for qubit in componentAdress:
            # term with single spin after applying QUBO to Ising transformation
            self.addInteraction(qubit, 0.5 * couplingStrength)
            # term with constant cost constribution after applying QUBO to Ising transformation
            self.addInteraction(0.5 * couplingStrength * self.data[qubit])


    def coupleComponents(self, firstComponent, secondComponent, couplingStrength=1, time=0):
        """given two lables, calculates the product of the corresponding qubits and
        translates it into an ising interaction"""
        # encode direction/sign of power flow in sign of coupling strength
        firstComponentAdress = self.getMemoryAdress(firstComponent,time)
        secondComponentAdress = self.getMemoryAdress(secondComponent,time)

        if firstComponentAdress[0] > secondComponentAdress[0]:
            self.coupleComponents(
                    secondComponent, firstComponent, couplingStrength, time=time
            )
            return

        for first in range(len(firstComponentAdress)):
            for second in range(len(secondComponentAdress)):
                # term with two spins after applying QUBO to Ising transformation
                # if both spins are the same, this will add a constant cost.
                # addInteraction performs substitution of spin with a constant
                self.addInteraction(
                        firstComponentAdress[first],
                        secondComponentAdress[second],
                        couplingStrength * 0.25
                )

                # terms with single spins after applying QUBO to Ising transformation
                self.addInteraction(
                        firstComponentAdress[first],
                        couplingStrength * self.data[secondComponent]['weights'][second] * 0.25
                )
                self.addInteraction(
                        secondComponentAdress[second],
                        couplingStrength * self.data[firstComponent]['weights'][first] * 0.25
                )

                # term with constant cost constribution after applying QUBO to Ising transformation
                self.addInteraction(
                    self.data[firstComponent]['weights'][first] * \
                    self.data[secondComponent]['weights'][second] * \
                    couplingStrength * 0.25
                )


    def encodeKirchhoffConstraint(self, bus):
        """
        add interactions to QUBO that enforce the Kirchhoff Constraint on solutions.
        This means that the total power at a bus is as close as possible to the load
        """
        components = self.getBusComponents(bus)
        flattenedComponenents = components['generators'] + \
                components['positiveLines'] + \
                components['negativeLines']

        demand = self.getLoad(bus)

        # constant load contribution to cost function
        self.addInteraction(demand ** 2)

        for component1 in flattenedComponenents:
            factor = 1.0
            if component1 in components['negativeLines']:
                factor *= -1.0

            # reward/penalty term for matching/adding load
            self.coupleComponentWithConstant(component1, - 2.0 * factor * demand)
            for component2 in flattenedComponenents:
                if component2 in components['negativeLines']:
                    curFactor = -factor
                else:
                    curFactor = factor

                # attraction/repulsion term for different/same sign of power at components
                self.coupleComponents(component1, component2, couplingStrength= curFactor)


    def siquanFormat(self):
        """Return the complete problem in the format for the siquan solver"""
        return [(v, list(k)) for k, v in self.problem.items() if v != 0 and len(k) > 0]


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
        # problemDict._marginalCosts()

        # for gen in problemDict._startIndex:
            # for t in range(len(problemDict.snapshots)):
                # problemDict._startupShutdownCost(gen, t)

        # kirchhoff constraints
        for node in problemDict.network.buses.index:
            for t in range(len(problemDict.snapshots)):

                problemDict.encodeKirchhoffConstraint(node)

        return problemDict


    # @staticmethod
    def addSQASolutionToNetwork(self, network, solutionState):
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

    def numVariables(self):
        """Return the number of spins."""
        return self.allocatedQubits

    def getValueToResult(self, component, result, time=0):
        value = 0
        encodingLength = self.data[component]["encodingLength"]
        for idx in range(time*encodingLength, (time+1)*encodingLength,1):
            if self.data[component]['indices'][idx] in result:
                value += self.data[component]['weights'][idx]
        return value


    def individualCostContribution(self, result):
        # TODO proper cost
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib, **self.calcKirchhoffCostAtBus(bus, result)}
        return contrib


    def calcKirchhoffCostAtBus(self, bus, result):
        contrib = {}
        for t in range(len(self.snapshots)):
            load = - self.getLoad(bus,t)
            components = self.getBusComponents(bus)
            for gen in components['generators']:
                load += self.getValueToResult(gen, result, time=t)
            for lineId in components['positiveLines']:
                load += self.getValueToResult(lineId, result, time=t)
            for lineId in components['negativeLines']:
                load -= self.getValueToResult(lineId, result, time=t)
            load = (load * self.kirchhoffFactor) ** 2
            contrib[str((bus, t))] = load
        return contrib


    def calcCost(self, result, addConstContribution=True):
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
        return totalCost


    # OLD CODE


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


    def marginalCosts(self):
        for gen in self.network.lines.index:
            self.biasComponent(
                    gen,
                    0.5 * self.monetaryCostFactor
            )

            generatorCost = self.network.generators["marginal_cost"].loc[gen] * \
                    self.getGeneratorOutput(gen, time)
            self.biasComponent(
                    gen,
                    0.5 * self.monetaryCostFactor * generatorCost,
            )

    def startupShutdownCost(self, gen, t):
        pass

    def getMaxLineEnergy(self, line):
        """Return the maximum value of energy a line can transport.
        """
        return self.network.lines.loc[line].s_nom




        for gen in self._startIndex:
            for t in range(len(self.snapshots)):
                # operating cost contribution
                index = self.toVecIndex(gen, t)
                val = self.network.generators["marginal_cost"].loc[gen] * (
                    self.getGeneratorOutput(gen, t)
                )
                self.addInteraction(index, self.monetaryCostFactor * val)


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

    def lineToIndex(self, lineId, time, binaryPos):
        return (
            self._lineIndices[lineId]
            + time * self.getBinaryLength(lineId)
            + binaryPos
        )

    def lineDirectionToIndex(self, lineId, time):
        return self._lineDirection[lineId] + time

    def lineSlackToIndex(self, lineId, time, binaryPos):
        return (
            self._slackVarsForQubo[lineId]
            + time * self.getBinaryLength(lineId)
            + binaryPos
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

        Currently we round this to the next highest power of 2 (-1)
        """
        return 2 ** self.getBinaryLength(line) - 1
        #return self.network.lines.loc[line].s_nom

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
            if load in self.network.loads_t.p_set:
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
            lenbinary1 = self.getBinaryLength(lineId1)
            for i in range(lenbinary1):
                for lineId2, factor2 in lineIdsFactor.items():
                    lenbinary2 = self.getBinaryLength(lineId2)
                    for j in range(lenbinary2):
                        # add the crossterm between line slack vars and line vars
                        # because the same loops are needed
                        self.addInteraction(
                            self.lineToIndex(lineId1, t, i),
                            self.lineSlackToIndex(lineId2, t, j),
                            -self.kirchhoffFactor
                            * 2 ** i
                            # * 2 ** j
                            * factor1
                            * factor2,
                        )
                        if lineId1 != lineId2 or i != j:
                            # cross term between total line sum and individual line
                            self.addInteraction(
                                self.lineToIndex(lineId1, t, i),
                                self.lineToIndex(lineId2, t, j),
                                self.kirchhoffFactor
                                / 4
                                * 2 ** i
                                * 2 ** j
                                * factor1
                                * factor2,
                            )
                            # slack vars make up the second quadratic sum
                            self.addInteraction(
                                self.lineSlackToIndex(lineId1, t, i),
                                self.lineSlackToIndex(lineId2, t, j),
                                self.kirchhoffFactor
                                # * 2 ** i
                                # * 2 ** j
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
                    self.lineSlackToIndex(lineId1, t, i),
                    self.kirchhoffFactor
                    * (maxGenPossible + maxLineCapacity + 2 * c_nt)
                    * factor1
                    # * 2 ** i,
                )
                self.addInteraction(
                    self.lineToIndex(lineId1, t, i),
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
                lenbinary1 = self.getBinaryLength(lineId1)
                for i in range(lenbinary1):
                    val = (
                        self.kirchhoffFactor
                        * self.getGeneratorOutput(gen0, t)
                        * factor1
                        # * 2 ** i
                    )
                    if index is None:
                        self.addInteraction(
                            self.lineToIndex(lineId1, t, i),
                            -0.5 * val,
                        )
                        self.addInteraction(
                            self.lineSlackToIndex(lineId1, t, i),
                            val,
                        )
                    else:
                        self.addInteraction(
                            index,
                            self.lineToIndex(lineId1, t, i),
                            -0.5 * val,
                        )
                        self.addInteraction(
                            index,
                            self.lineSlackToIndex(lineId1, t, i),
                            val,
                        )

    def _addSlackConstraints(self):
        for index in self.network.lines.index:
            lenbinary = self.getBinaryLength(index)
            for t in range(len(self.network.snapshots)):
                for i in range(lenbinary):
                    self.addInteraction(
                        self.lineToIndex(index, t, i),
                        -self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineDirectionToIndex(index, t),
                        -self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineSlackToIndex(index, t, i),
                        2 * self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineToIndex(index, t, i),
                        self.lineDirectionToIndex(index, t),
                        self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineToIndex(index, t, i),
                        self.lineSlackToIndex(index, t, i),
                        -2 * self.slackVarFactor,
                    )
                    self.addInteraction(
                        self.lineDirectionToIndex(index, t),
                        self.lineSlackToIndex(index, t, i),
                        -2 * self.slackVarFactor,
                    )

    def constantSlackCost(self):
        totalConstCost = 0.0
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

