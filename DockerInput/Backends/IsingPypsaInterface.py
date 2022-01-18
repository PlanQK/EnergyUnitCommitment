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
        # contains qubo coefficients
        self.problem = {}
        self.snapshots = snapshots

        self.allocatedQubits = 0
        # contains encoding data
        self.data = {}

        envMgr = EnvironmentVariableManager()
        self.kirchhoffFactor = float(envMgr["kirchhoffFactor"])
        self.monetaryCostFactor = float(envMgr["monetaryCostFactor"])
        self.minUpDownFactor = float(envMgr["minUpDownFactor"])
        self.slackVarFactor = float(envMgr["slackVarFactor"])

        lineRepresentation = envMgr["lineRepresentation"]
        self.maxOrder = int(envMgr["maxOrder"])

        if lineRepresentation == "":
            self.network = network
        else:
            self.lineRepresentation = int(lineRepresentation)
            self.network = network.copy()
            


        self.storeGenerators()

        self.storeLines()

        self.storeSlackVars()

        
    def buildQubo(self):
        pass
        



    def storeGenerators(self):
        time = 0
        for i in range(len(self.network.buses)):
            generatorsAtBus = self.network.generators[
                self.network.generators.bus == self.network.buses.index[i]
            ]
            for gen in generatorsAtBus.index:
                if self.network.generators.committable[gen]:
                    continue
                self.data[gen] = {
                        'indices' : [self.allocatedQubits],
                        'values' : [self.network.generators_t.p_max_pu[gen].iloc[time]]
                }
                self.allocatedQubits += len(self.snapshots)
        return


    def totalLoad(self, bus, time):
        if time is None:
            time = self.network.loads_t['p_set'].index[0]
        loadsAtCurrentBus = self.network.loads[
                                    self.network.loads.bus == bus
                            ].index
        allLoads = self.network.loads_t['p_set'].loc[time]

        result = allLoads[allLoads.index.isin(loadsAtCurrentBus)]
        return result.sum()

    def getTotalEncodedValue(self, component):
        result = 0
        for val in self.data[component]['values']:
            result += val
        return result

    def getBusComponents(self, bus):
        """return all labels of components that connect to a bus as a dictionary
        generators - at this bus
        positiveLines - start in this bus
        negativeLines - end in this bus
        loads - at this bus
        """
        result = {
                "generators":
                        list(self.network.generators[
                                self.network.generators.bus == bus
                        ].index)
                    ,
                "positiveLines" :
                        list(self.network.lines[
                                self.network.lines.bus1 == bus
                        ].index)
                ,
                "negativeLines" :
                        list(self.network.lines[
                                self.network.lines.bus0 == bus
                        ].index)
                ,
                }
        return result

        

    def storeLines(self):
        "calls encodeLine for each line"
        for index in self.network.lines.index:
            # overwrite this in child classes
            self.encodeLine(index)

    def encodeLine(self, line):
        """
        split lines into 1-valued qubits
        """
        
        capacity = int(self.network.lines.loc[line].s_nom)
        indices = list(range(self.allocatedQubits, self.allocatedQubits+ 2*capacity,1) )
        values = [1 for _ in range(0,capacity,1)] + [-1 for _ in range(0,capacity,1)]

        self.allocatedQubits += len(indices)

        self.data[line] = {
                'values' : values,
                'indices' : indices,
        }
        return

    def getLineValue(self, line, solution):
        result = 0
        for idx in range(len(self.data[line]['indices'])):
            if self.data[line]['indices'][idx] in solution:
                result+=self.data[line]['values'][idx]
        return result

    def getLineValues(self, solution):
        solution = set(solution)
        lineValues = {}
        for lineId in self.network.lines.index:
            for t in range(len(self.snapshots)):
                value = self.getLineValue(lineId, solution)
                lineValues[(lineId, t)] = value
        return lineValues

    def storeSingleton(self, label, value):
        if hasattr(self.data, label):
            raise ValueError("The label already exists")
        self.data[label] = {
                'values': [value],
                'indices': [self.allocatedQubits],
                }
        self.allocatedQubits += 1

    def storeSlackVars(self):
        pass



    def coupleComponents(self, firstComponent, secondComponent, couplingStrength=1, additive = True):
        "given two labels couples all qubits in their representation"
        # TODO: with or without scaling factor?"
        firstComponentAdress = self.getMemoryAdress(firstComponent)
        secondComponentAdress = self.getMemoryAdress(secondComponent)
        
        if firstComponentAdress[0] > secondComponentAdress[0]:
            self.coupleComponents(secondComponent, firstComponent, couplingStrength, additive=additive)
            return

        for first in range(len(firstComponentAdress)):
            for second in range(len(secondComponentAdress)):
                key = (firstComponentAdress[first], secondComponentAdress[second])
                if key[0] == key[1]:
                    # continue
                    key = (key  [0],)
                interactionStrength = - couplingStrength * \
                        self.data[firstComponent]['values'][first] * \
                        self.data[secondComponent]['values'][second]

                if additive:
                    interactionStrength += self.problem.get(key,0)
                self.problem[key] = interactionStrength
            

    def biasComponent(self, component,bias=1, additive = True, ):
        "given a component level, sets the bias for all qubits in their representation"
        componentAdress = self.getMemoryAdress(component)
        for idx in range(len(componentAdress)):
            key = (componentAdress[idx],)
            interactionStrength = bias * \
                    self.data[component]['values'][idx] 
            if additive:
                interactionStrength += self.problem.get(key,0)
            self.problem[key] = interactionStrength



    def getMemoryAdress(self, component):
        return self.data[component]["indices"]
        

        

    def encodeKirchhoffConstraint(self, bus):
        """
        add interactions to QUBO that enforce the Kirchhoff Constraint on solutions.
        This means that the total power at a bus is as close as possible to the load
        """

        components = self.getBusComponents(bus)
        flattenedComponenents = components['generators'] + \
                components['positiveLines'] + \
                components['negativeLines']

        demand = self.totalLoad(bus,time=None)
        print(f"{bus} :: Demand :: {demand}")

        for component1 in flattenedComponenents:
            factor = 1.0

            # self.biasComponent(component1, bias = 1)
            if component1 in components['negativeLines']:
                print(f"negativeLine :: {component1}")
                factor *= -1.0
            self.biasComponent(component1, bias= -factor * 0.5 * demand)

            for component2 in flattenedComponenents:
                if component2 in components['negativeLines']:
                    curFactor = -factor
                else:
                    curFactor = factor

                self.coupleComponents(component1, component2, couplingStrength= factor * 0.25)
                self.biasComponent(component1, bias= 0.25 * curFactor * 1)
                self.biasComponent(component2, bias= 0.25 * curFactor * 1)
        print(self.problem)
        print(f"ending {bus}")


    def siquanFormat(self):
        """Return the complete problem in the format for the siquan solver"""
        return [(v, list(k)) for k, v in self.problem.items()]


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

    def numVariables(self):
        """Return the number of spins."""
        return self.allocatedQubits

    def getValueToResult(self, component, result):
        value = 0 
        for idx in range(len(self.data[component]['indices'])):
            if self.data[component]['indices'][idx] in result:
                value += self.data[component]['values'][idx]
        return value


    def individualCostContribution(self, result):
        # TODO proper cost
        contrib = {}
        for node in self.network.buses.index:
            for t in range(len(self.snapshots)):
                load = - self.totalLoad(node,time=None)
                components = self.getBusComponents(node)
                for gen in components['generators']:
                    load += self.getValueToResult(gen, result)
                for lineId in components['positiveLines']:
                    load += self.getValueToResult(lineId, result)
                for lineId in components['negativeLines']:
                    load -= self.getValueToResult(lineId, result)
                contrib[str((node, t))] = load
        return contrib

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
        if addConstContribution:
            for node in self.network.buses.index:
                for t in range(len(self.snapshots)):
                    totalCost += self.constantCostContribution(node, t)
            totalCost += self.constantSlackCost()
        return totalCost

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

    def constantCostContribution(self, node, t):
        totalConstCost = 0
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
        )
        # these parts are from the quadratic sum where the gen, line, slacks are squared
        for gen0 in self.network.generators[
            self.network.generators.bus == node
        ].index:
            totalConstCost += (
                0.25
                * self.kirchhoffFactor
                * self.getGeneratorOutput(gen0, t) ** 2
            )

        for line in self.getLineFactors(node):
            for i in range(self.getBinaryLength(line)):
                totalConstCost += 1.25 * self.kirchhoffFactor * 2 ** (i * 2)
        return totalConstCost

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

    def splitLine(self, line, network, lineRepresentation=0, maxOrder=0):
        """
        splits up a line into multiple lines such that each new line
        has capacity 2^n - 1 for some n. Modifies the network given
        as an argument and returns the original line name and how
        many components where used to split it up. Use it on the
        network stored in self because it modifies it.

        lineRepresentation is an upper limit for number of splits. if
        is is 0, no limit is set
        """
        remaining_s_nom = network.lines.loc[line].s_nom
        numComponents = 0
        maxMagnitude = 2 ** self.maxOrder - 1

        while remaining_s_nom > 0 \
                and (lineRepresentation == 0 or lineRepresentation > numComponents):
            binLength = len("{0:b}".format(1+int(np.round(remaining_s_nom))))-1
            magnitude = 2 ** binLength - 1
            if maxOrder:
                magnitude = min(magnitude,maxMagnitude)
            remaining_s_nom -= magnitude
            network.add(
                "Line",
                f"{line}_split_{numComponents}",
                bus0=network.lines.loc[line].bus0,
                bus1=network.lines.loc[line].bus1,
                s_nom=magnitude
            )
            numComponents += 1

        network.remove("Line",line)
        return (line, numComponents)


    def mergeLines(self, lineValues, snapshots):
        """
        For a dictionary of lineValues of the network, whose
        lines were split up, uses the data in self to calculate
        the corresponding lineValues in the unmodified network
        """
        if hasattr(self, 'lineDictionary'):
            result = {}
            for line, numSplits in self.lineDictionary.items():
                for snapshot in range(len(snapshots)):
                    newKey = (line, snapshot)
                    value = 0
                    for i in range(numSplits):
                        splitLineKey = line + "_split_" + str(i)
                        value += lineValues[(splitLineKey, snapshot)]
                    result[newKey] = value
            return result
        else:
            return lineValues
    

