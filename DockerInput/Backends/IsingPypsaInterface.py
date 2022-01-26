from collections import OrderedDict
import numpy as np
from EnvironmentVariableManager import EnvironmentVariableManager
import typing


class IsingPypsaInterface:

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
        if result == 0:
            print(f"Warning: No load at {bus} at timestep {time}")
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
        it into an ising interaction. This uses QUBO formulation"""
        componentAdress = self.getMemoryAdress(component,time)
        for qubit in componentAdress:
            # term with single spin after applying QUBO to Ising transformation
            self.addInteraction(qubit, 0.5 * couplingStrength)
            # term with constant cost constribution after applying QUBO to Ising transformation
            self.addInteraction(0.5 * couplingStrength * self.data[qubit])


    def coupleComponents(self, firstComponent, secondComponent, couplingStrength=1, time=0, additionalTime=None):
        """given two lables, calculates the product of the corresponding qubits and
        translates it into an ising interaction. This uses QUBO formulation"""

        if additionalTime is None:
            additionalTime = time

        firstComponentAdress = self.getMemoryAdress(firstComponent,time)
        secondComponentAdress = self.getMemoryAdress(secondComponent,additionalTime)

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


    def encodeKirchhoffConstraint(self, bus, time=0):
        """
        add interactions to QUBO that enforce the Kirchhoff Constraint on solutions.
        This means that the total power at a bus is as close as possible to the load
        """
        components = self.getBusComponents(bus)
        flattenedComponenents = components['generators'] + \
                components['positiveLines'] + \
                components['negativeLines']

        demand = self.getLoad(bus, time=time)

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
                self.coupleComponents(component1, component2, couplingStrength=curFactor)

    def encodeStartupShutdownCost(self, bus, time=0):
        """
        encode cost for shutting down or starting a generator at timestep t. 
        This requires t > 0 so there is a status to be changed in the previous
        time step"""

        # no previous information on first time step or when out of bounds
        if time == 0 or time >= len(self.snapshots):
            return

        generators = self.getBusComponents(bus)['generators']

        for generator in generators:

            startup_cost = self.network.generators["start_up_cost"].loc[generator]
            shutdown_cost = self.network.generators["shut_down_cost"].loc[generator]

            # start up costs
            # summands of (1-g_{time-1})  * g_{time})
            self.coupleComponentWithConstant(
                    generator,
                    couplingStrength=self.monetaryCostFactor * startup_cost,
                    time=time
            )
            self.coupleComponents(
                    generator,
                    generator,
                    couplingStrength= -self.monetaryCostFactor * startup_cost,
                    time = time,
                    additionalTime = time -1
            )

            # shutdown costs
            # summands of g_{time-1} * (1-g_{time})
            self.coupleComponentWithConstant(
                    generator,
                    couplingStrength=self.monetaryCostFactor * shutdown_cost,
                    time=time-1
            )
            self.coupleComponents(
                    generator,
                    generator,
                    couplingStrength= -self.monetaryCostFactor * shutdown_cost,
                    time = time,
                    additionalTime = time -1
            )






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
        for time in range(len(problemDict.snapshots)):
            for node in problemDict.network.buses.index:
                problemDict.encodeMarginalCosts(node,time)
                problemDict.encodeKirchhoffConstraint(node,time)
                problemDict.encodeStartupShutdownCost(node,time)

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

    def marginalCostOffset(self, time=0):
        """
        returns a float by which all generator marginal costs per power will be offset.
        Since every generator will be offset, this will not change relative costs between them
        It changes the range of energy contributions this constraint provides. Adding marginal
        costs as a cost to the QUBO formulation will penalize all generator configurations. The offset
        shifts it so that better than average generator configuration will be rewarded.
        """
        # reward better than average configurations and penalize worse than average configurations
        return self.estimateAverageCostPerPowerGenerated(time)


    def estimateAverageCostPerPowerGenerated(self, time=0):
        """
        calculates average cost power power unit produced if all generators
        were switched on. returns a float, not a numpy.float
        """
        maxCost = 0.0
        maxPower = 0.0
        for generator in self.network.generators.index:
            currentPower = self.network.generators_t.p_max_pu[generator].iloc[time]
            maxCost += currentPower * self.network.generators["marginal_cost"].loc[generator]
            maxPower += currentPower
        try:
            return maxCost / maxPower
        except ZeroDivisionError:
            print(f"No available Power at timestep: {time}")
            raise ZeroDivisionError

    def encodeMarginalCosts(self, bus, time):
        """encodes marginal costs for running generators and transmission lines at a single bus.
        This uses an offset calculated in ´marginalCostOffset´, which is a dependent on all generators
        of the entire network for a single time slice"""

        components = self.getBusComponents(bus)
        costOffset = self.marginalCostOffset(time)
        for generator in components['generators']:
            self.coupleComponentWithConstant(
                    generator, 
                    couplingStrength=(self.network.generators["marginal_cost"].loc[generator] - costOffset) * \
                            self.monetaryCostFactor,
                    time=time
            )

        for line in components['positiveLines']:
            # TODO What is the cost of transferring power?
            pass
            
