from collections import OrderedDict
import numpy as np
from EnvironmentVariableManager import EnvironmentVariableManager
import typing


class IsingPypsaInterface:
    """
    class to generate and store an Ising spin glass problem.

    @attribute network: pypsa.Network
        the network for which to build in ising spin glass problem for
    @attribute snapshots: list
        list of snapshots to be considered in problem
    @allocatedQubits: int
        number of currently used qubits
    @attribute data: dict
        a dictionary that stores all encoding with qubits related data 
        for the network components. 
        @key: int
            weight of the qubit
        @key: str
            data as dictionary corresponding to the component label
    @attribute kirchhoffFactor: float
        weight of kirchhoff constraint
    @attribute monetaryCostFactor: float
        weight of any monetary cost incurred by a solution
    @attribute minUpDownFactor: float
        weight of minimal up/down-time constraint
    """

    @classmethod
    def buildCostFunction(
        cls,
        network,
    ):
        """
        factory method to instantiate a child class of IsingPypsaInterface. Additional
        parameters to determine the appropiate child class are read from the environment.

        @param network: pypsa.Network
            A pypsa network for which to formulate the unit commitment problem as an
            ising spin glass problem
        @return: IsingPypsaInterface
            instance of IsingPypsaInterface child class with complete problem formulation
        """

        envMgr = EnvironmentVariableManager()

        problemFormulation = envMgr["problemFormulation"]
        if problemFormulation == "fullsplit":
            IsingObject = fullsplitIsingInterface(network, network.snapshots)

        return IsingObject

    def __init__(self, network, snapshots):
        """
        Constructor for an IsingPypsaInterface. It reads all relevant parameters
        for a problem formulation from the environment and instantiates all attributes
        that other class method write in and read from. It does not fill any of those
        attributes with data specific to a chosen problem formulation

        @param network: pypsa.Network
            The pypsa.Network for which to build an Ising spin glass problem
        @param snapshots: list
            integer indices of snapshots to consider in Ising spin glass problem
        @return: IsingPypsaInterface
            An empty IsingPypsaInterface object
        """

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


    def writeToHighestLevel(self, component):
        """
        After storing all qubits that represent a logical component of the network
        (generators, lines) this writes the weight of all used qubits i into the
        data dictionary at the highest level for access as self.data[i]

        @param component: str
            the label of a network component
        @return: None
            modifies the dictionary self.data 
        """
        for idx in range(len(self.data[component]['indices'])):
            self.data[self.data[component]['indices'][idx]] = self.data[component]['weights'][idx]


    def storeGenerators(self):
        """
        Assigns qubits (int) to each generator in self.network. For each generator it writes
        generator specific parameters(power, corresponding qubits, size of encoding) into 
        the dictionary self.data. At last it updates object specific parameters

        @return: None
            modifies self.data and self.allocatedQubits
        """

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
        on all lines at each time slice
        
        @return: None
            modifies self.data and self.allocatedQubits
        """
        for line in self.network.lines.index:
            for time in range(len(self.network.snapshots)):
            # overwrite this in child classes
                self.encodeLine(line,time)
            self.writeToHighestLevel(line)

    def encodeLine(self, line, time):
        """
        Allocate qubits to encode a line at a single time slice. The specific encoding
        of the line is determined by the method "splitCapacity". Other encodings can be
        obtained by overwriting "splitCapacity" in a child class.

        @param line: str
            label of the network line to be encoded in qubits
        @param time: int
            index of time slice at which to encode the line
        @return: None
            modifies self.allocatedQubits and self.data

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

    # @abstractmethod
    def splitCapacity(self, capacity):
        """
        Method to split a line which has maximum capacity "capacity". A line split is a 
        list of lines with varying capacity which can either be on or off. The status of 
        a line is binary, so it can either carry no power or power equal to it's capacity
        The direction of flow for each line is also fixed. It is not enforced that a
        chosen split can only represent flow lower than capacity or all flows that are
        admissable. 

        @param capacity: int
            the capacity of the line that is to be split up
        @return: list
            a list of integers. Each integer is the capacity of a line of the splitting
            direciont of flow is encoded as the sign
        """
        raise NotImplementedError("No implementation for splitting up a line into multilple components")

    # ------------------------------------------------------------------------
    # helper functions to obtain represented values

    def getBusComponents(self, bus):
        """
        Returns all labels of components that connect to a bus as a dictionary. 
        For lines that end in this bus, positive power flow is interpreted as
        increasing available power at the bus. For Lines that start in this bus
        positive power flow is interpreted as decreasing available power at the bus.

        @param bus: str
            label of the bus
        @return: dict
            @key 'generators'
                list of labels of generators that are at the bus
            @key 'positiveLines'
                list of labels of lines that start in this bus
            @key 'negativeLines'
                list of labels of lines that end in this bus
         - end in this bus
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
        """
        return the status of a generator(on, off) in a given solution

        @param gen: str
            label of the generator
        @param solution: list
            list of all qubits which have spin -1 in the solution
        @param time: time
            index of time slice for which to get the generator status
        """
        return self.data[gen]['indices'][time] in solution

    def getFlowDictionary(self, solution):
        """
        builds a dictionary contain all power flows at all time slices for a given
        solution of qubit spins

        @param solution: list
           list of all qubits which have spin -1 in the solution 
        @return: dict
            @key: (str,int)
                label of line and index of time slice
        """
        solution = set(solution)
        result = {}
        for lineId in self.network.lines.index:
            for time in range(len(self.snapshots)):
                result[(lineId, time)] = self.getEncodedValueOfComponent(lineId, solution, time)
        return result


    def getLineValues(self, solution):
        """
        wrapper for calling getFlowDictionary. It builds a dictionary that contains
        all power flows at all time slices for a given solution of qubit spins

        @param solution: list
           list of all qubits which have spin -1 in the solution 
        @return: dict
            @key: (str,int)
                label of line and index of time slice
        """
        return self.getFlowDictionary(solution)


    def getLoad(self, bus, time=0):
        """
        returns the total load at a bus at a given time slice

        @param bus: str
            label of bus at which to calculate the total load
        @param time: int
            index of time slice for which to get the total load
        """
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
        """
        Returns a list of all qubits that are used to encode a network component
        at a given time slice. A component is assumed to be encoded in one block
        with constant encoding size per time slice and order of time slices
        being respected in the encoding

        @param component: str
            label of the network component
        @param time: int
            index of time slice for which to get representing qubits
        @return: list
            list of integers which are qubits that represent the component
        """
        encodingLength = self.data[component]["encodingLength"]
        return self.data[component]["indices"][time * encodingLength : (time+1) * encodingLength]


    def siquanFormat(self):
        """
        Return the complete problem in the format for the siquan solver
        
        @return: list
            list of tuples of the form (interaction-coefficient, list(qubits))
        """
        return [(v, list(k)) for k, v in self.problem.items() if v != 0 and len(k) > 0]



    # TODO
    # @staticmethod
    def addSQASolutionToNetwork(self, network, solutionState):
        """
        writes the solution encoded in an ising spin glass problem into the 
        pypsa network
        
        @param network: pypsa.Network
            the pypsa network in which to write the results
        @param solutionState: list
            list of all qubits which have spin -1 in the solution 
        @return: None
            modifies network changing generator status and power flows
        
        """
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
        """
        Return the number of currently used qubits

        @return: int
            number of qubits in use
        """
        return self.allocatedQubits

    def getEncodedValueOfComponent(self, component, result, time=0):
        """
        Returns the encoded value of a component according to the spin configuration in result
        at a given time slice

        @param component: str
            label of the network component for which to retrieve encoded value
        @param result: list
            list of all qubits which have spin -1 in the solution
        @param time: int
            index of time slice for which to retrieve encoded value
        @return: float
            value of component encoded in the spin configuration of result
        """
        value = 0.0
        encodingLength = self.data[component]["encodingLength"]
        for idx in range(time*encodingLength, (time+1)*encodingLength,1):
            if self.data[component]['indices'][idx] in result:
                value += self.data[component]['weights'][idx]
        return value

    def calcKirchhoffCostAtBus(self, bus, result):
        """
        returns a dictionary which contains the kirchhoff cost at the specified bus 'bus' for
        every time slice 'time' as {(bus,time) : value} 

        @param result: list
           list of all qubits which have spin -1 in the solution 
        @return: dict
            dictionary with keys of the type (str,int) over all  time slices and the string 
            alwyays being the chosen bus
        """
        contrib = {}
        for t in range(len(self.snapshots)):
            load = - self.getLoad(bus,t)
            components = self.getBusComponents(bus)
            for gen in components['generators']:
                load += self.getEncodedValueOfComponent(gen, result, time=t)
            for lineId in components['positiveLines']:
                load += self.getEncodedValueOfComponent(lineId, result, time=t)
            for lineId in components['negativeLines']:
                load -= self.getEncodedValueOfComponent(lineId, result, time=t)
            load = (load * self.kirchhoffFactor) ** 2
            contrib[str((bus, t))] = load
        return contrib

    def individualCostContribution(self, result):
        """
        returns a dictionary which contains the kirchhoff cost incurred at every bus at
        every time slice

        @param result: list
           list of all qubits which have spin -1 in the solution 
        @return: dict
            dictionary with keys of the form (str,int) over all busses and time slices
        """
        # TODO proper cost
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib, **self.calcKirchhoffCostAtBus(bus, result)}
        return contrib

    def individualMarginalCost(self, result):
        """
        returns a dictionary which contains the marginal cost incurred at every bus 'bus' at
        every time slice 'time' as {(bus,time) : value} 

        @param result: list
           list of all qubits which have spin -1 in the solution 
        @return: dict
            dictionary with keys of the type (str,int) over all busses and time slices
        """
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib, **self.calcMarginalCostAtBus(bus, result)}
        return contrib

    def calcMarginalCostAtBus(self, bus, result):
        """
        returns a dictionary which contains the marginal cost the specified bus 'bus' at
        every time slice 'time' as {(bus,time) : value} 

        @param result: list
           list of all qubits which have spin -1 in the solution 
        @return: dict
            dictionary with keys of the type (str,int) over all  time slices and the string 
            alwyays being the chosen bus
        """
        contrib = {}
        for time in range(len(self.snapshots)):
            marginalCost = 0.0
            components = self.getBusComponents(bus)
            for generator in components['generators']:
                if self.getGeneratorStatus(generator, result, time):
                    marginalCost += self.network.generators["marginal_cost"].loc[generator]
            contrib[str((bus, time))] = marginalCost

        return contrib

    def calcMarginalCost(self, result):
        """
        calculate the total marginal cost incurred by a solution

        @param result: list
            list of all qubits which have spin -1 in the solution
        @return: float
            total marginal cost incurred without monetaryFactor scaling
        """
        marginalCost = 0.0
        for key, val in self.individualMarginalCost(result).items():
            marginalCost += val 
        return marginalCost


    def calcCost(self, result, addConstContribution=True):
        """
        calculates the energy of a spin state
        
        @param result: list
            list of all qubits which have spin -1 in the solution 
        @return: float
            the energy of the spin glass state in result
        
        """
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


    # ------------------------------------------------------------
    # functions to couple components. The couplings are interpreted as multiplications of QUBO
    # polynomials. The final interactions are coefficients for an ising spin glass problem

    def addInteraction(self, *args):
        """
        Helper function to define an Ising Interaction. The interaction is scaled by all qubit
        specific weights. For higher order interactions, it performs substitutions of qubits
        that occur multiple times, which would be constant in an ising spin glass problem.
        Interactions are stored in the attribute "problem", which is a dictionary
        Keys are tupels of involved qubits and values are floats

        The method can take an arbitrary number of arguments:
        The last argument is the interaction strength.
        The previous arguments contain the spin ids.

        @param args[-1]: float
            the basic interaction strength before appling qubit weights
        @param args[:-1]: list
            list of all qubits that are involved in this interaction
        @return: None
            modifies self.problem by adding the strength of the interaction if an interaction
            coefficient is already set
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
        # makes it into a constant contribution. Doesn't work properly for higer order interactions
        if len(key) == 2:
            if key[0] == key[1]:
                key = tuple([])
        self.problem[key] = self.problem.get(key,0) - interactionStrength


    def coupleComponentWithConstant(self, component, couplingStrength=1, time=0):
        """
        Performs a QUBO multiplication involving a single variable on all qubits which are logically
        grouped to represent a component at a given time slice. This QUBO multiplication is
        translated into Ising interactions and then added to the currently stored ising spin glass
        problem

        @param component: str
            label of the network component
        @param couplingStrength: float
            cofficient of QUBO multiplication by which to scale the interaction. Does not contain 
            qubit specific weight
        @param time: int
            index of time slice for which to couple qubit representing the component
        @return: None
            modifies self.problem. Adds to previously written interaction cofficient
        """
        componentAdress = self.getMemoryAdress(component,time)
        for qubit in componentAdress:
            # term with single spin after applying QUBO to Ising transformation
            self.addInteraction(qubit, 0.5 * couplingStrength)
            # term with constant cost constribution after applying QUBO to Ising transformation
            self.addInteraction(0.5 * couplingStrength * self.data[qubit])


    def coupleComponents(self, firstComponent, secondComponent, couplingStrength=1, time=0, additionalTime=None):
        """
        Performs a QUBO multiplication involving exactly two components on all qubits which are logically
        grouped to represent these components at a given time slice. This QUBO multiplication is
        translated into Ising interactions and then added to the currently stored ising spin glass
        problem

        @param firstComponent: str
            label of the first network component
        @param secondComponent: str
            label of the second network component
        @param couplingStrength: float
            cofficient of QUBO multiplication by which to scale the interaction. Does not contain 
            qubit specific weights
        @param time: int
            index of time slice of the first component for which to couple qubit representing it
        @param additionalTime: int
            index of time slice of the second component for which to couple qubit representing it.
            The default parameter None is used if the time slices of both components are the same 
        @return: None
            modifies self.problem. Adds to previously written interaction cofficient

        @example:
            Let X_1, X_2 be the qubits representing firstComponent and Y_1, Y_2 the qubits representing
            secondComponent. The QUBO product the method translates into ising spin glass coefficients is:
            (X_1 + X_2) (Y_1 + Y_2) = X_1 Y_1 + X_1 Y_2 + X_2 Y_1 + X_2 Y_2
        """

        if additionalTime is None:
            additionalTime = time

        firstComponentAdress = self.getMemoryAdress(firstComponent,time)
        secondComponentAdress = self.getMemoryAdress(secondComponent,additionalTime)

        # components with 0 weight (power, capacity) vanish in the QUBO formulation
        if (not firstComponentAdress) or (not secondComponentAdress):
            return

        # order adress
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


    # ------------------------------------------------------------
    # encodings of problem constraints

    def encodeKirchhoffConstraint(self, bus, time=0):
        """
        Adds the kirchhoff constraint at a bus to the problem formulation. The kirchhoff constraint
        is that the sum of all power generating elements (generators, lines ) is equal to the sum of 
        all load generating elements (bus specific load, lines). Deviation from equality is penalized
        quadratically 

        @param bus: str
            label of the bus at which to enforce the kirchhoff constraint
        @param time: int
            index of time slice at which to enforce the kirchhoff contraint
        @return: None
            modifies self.problem. Adds to previously written interaction cofficient
        """
        components = self.getBusComponents(bus)
        flattenedComponenents = components['generators'] + \
                components['positiveLines'] + \
                components['negativeLines']

        demand = self.getLoad(bus, time=time)

        # constant load contribution to cost function so that a configuration that fulfills the
        # kirchhoff contraint has energy 0
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
        Adds the startup and shutdown costs for every generator attached to the bus. Those
        costs are monetary costs incurred whenever a generator changes its status from one
        time slice to the next. The first time slice doesn't incurr costs because the status
        of the generators before is unknown
        
        @param bus: str
            label of the bus at which to add startup and shutdown cost
        @param time: int
            index of time slice which contains the generator status after a status change
        @return: None
            modifies self.problem. Adds to previously written interaction cofficient 
        
        """

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


    def marginalCostOffset(self, time=0):
        """
        returns a float by which all generator marginal costs per power will be offset.
        Since every generator will be offset, this will not change relative costs between them
        It changes the range of energy contributions this constraint provides. Adding marginal
        costs as a cost to the QUBO formulation will penalize all generator configurations. The offset
        shifts it so that better than average generator configuration will be rewarded.
        """
        # reward better than average configurations and penalize worse than average configurations
        return 0.0
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

    def calcEffiencyLoss(self, cheapGen, expensiveGen,time=0):
        """calculates an approximation of the loss of using a generator with higher operational 
        costs (expensiveGen) compared to using the generator with a lower cost (cheapGen)
        """
        cheapCost = self.network.generators["marginal_cost"].loc[cheapGen]
        expensiveCost = self.network.generators["marginal_cost"].loc[expensiveGen]
        matchedPower = min(
                self.data[cheapGen]['weights'][time],
                self.data[expensiveGen]['weights'][time],
                )
        if expensiveCost <= cheapCost:
            return 0
        result = (expensiveCost - cheapCost ) * self.monetaryCostFactor 
        return result


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


class fullsplitIsingInterface(IsingPypsaInterface):

    def __init__(self, network, snapshots):
        super().__init__(network, snapshots)

        # read generators and lines from network and encode as qubits
        self.storeGenerators()
        self.storeLines()

        # kirchhoff constraints
        for time in range(len(self.snapshots)):
            for node in self.network.buses.index:
                self.encodeMarginalCosts(node,time)
                self.encodeKirchhoffConstraint(node,time)
                self.encodeStartupShutdownCost(node,time)


    def splitCapacity(self, capacity):
        """
        Method to split a line which has maximum capacity "capacity". A line split is a 
        list of lines with varying capacity which can either be on or off. The status of 
        a line is binary, so it can either carry no power or power equal to it's capacity
        The direction of flow for each line is also fixed. It is not enforced that a
        chosen split can only represent flow lower than capacity or all flows that are
        admissable. 

        @param capacity: int
            the capacity of the line that is to be split up
        @return: list
            a list of integers. Each integer is the capacity of a line of the splitting
            direciont of flow is encoded as the sign
        """
        return [1 for _ in range(0,capacity,1)]  + [-1 for _ in range(0,capacity,1)]
