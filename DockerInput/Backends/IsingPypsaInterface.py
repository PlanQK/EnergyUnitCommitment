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
    @attribute allocatedQubits: int
        number of currently used qubits
    @attribute problem: dict
        dictionary that stores ising spin glass interactions
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

    def __init__(self, network, snapshots):
        """
        Constructor for an IsingPypsaInterface. It reads all relevant parameters
        for a problem formulation from the environment and instantiates all attributes
        that other class method write in and read from. It does not fill any of those
        attributes with data specific to a chosen problem formulation like component
        - qubit representations or problem contraint interactions

        @param network: pypsa.Network
            The pypsa.Network for which to build an Ising spin glass problem
        @param snapshots: list
            integer indices of snapshots to consider in Ising spin glass problem
        @return: IsingPypsaInterface
            An empty IsingPypsaInterface object
        """

        # network to be solved
        self.network = network
        self.snapshots = snapshots

        # hyper parameters of problem formulation
        envMgr = EnvironmentVariableManager()
        self.kirchhoffFactor = float(envMgr["kirchhoffFactor"])
        self.monetaryCostFactor = float(envMgr["monetaryCostFactor"])
        self.minUpDownFactor = float(envMgr["minUpDownFactor"])

        # contains ising coefficients
        self.problem = {}

        # contains encoding data
        self.data = {}
        
        # qubits currently in use/next qubit to use to represent a network component
        self.allocatedQubits = 0


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
        FactoryDictionary = {
                "fullsplit" : fullsplitNoMarginalCost,
                "fullsplitGlobalCostSquare" : fullsplitGlobalCostSquare,
                "fullsplitMarginalAsPenalty" : fullsplitMarginalAsPenalty,
                "fullsplitNoMarginalCost" : fullsplitNoMarginalCost,
                "fullsplitLocalMarginalEstimationDistance" : fullsplitLocalMarginalEstimationDistance,
                "fullsplitDirectInefficiencyPenalty" : fullsplitDirectInefficiencyPenalty,
                "binarysplitIsingInterface" : binarysplitIsingInterface,
                "binarysplitNoMarginalCost" : binarysplitNoMarginalCost,
                "fullsplitMarginalAsPenaltyAverageOffset" : fullsplitMarginalAsPenaltyAverageOffset,
        }
        return FactoryDictionary[problemFormulation](network, network.snapshots)


    def numVariables(self, ):
        return self.allocatedQubits


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


    def encodeGenerator(self, generator, time):
        """
        Allocate qubits to encode a generator at a single time splice. THe specific encoding
        for this method is that a generator's is assumed to be binary and either supplying
        full power or no power.
        
        Args:
            bus: (str) label of the generator to be encoded in qubits
            time: (int) index of time slice for which to encode the generator
        Returns:
            (None) modifies self.allocatedQubits and self.data
        """
        # no generator is supposed to be committable in our problems
        if self.network.generators.committable[generator]:
            return
        weights = [self.getNominalPower(generator, time)]
        indices = range(self.allocatedQubits, self.allocatedQubits + len(weights))
        self.allocatedQubits += len(indices)
        self.data[generator] = {
                'indices' : indices,
                'weights' : weights,
                'encodingLength' : len(weights),
        }
        return
    

    def storeGenerators(self):
        """
        Assigns qubits (int) to each generator in self.network. For each generator it writes
        generator specific parameters(power, corresponding qubits, size of encoding) into 
        the dictionary self.data. At last it updates object specific parameters

        @return: None
            modifies self.data and self.allocatedQubits
        """
        for generator in self.network.generators.index:
            for time in range(len(self.network.snapshots)):
                self.encodeGenerator(generator, time)
            self.writeToHighestLevel(generator)
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
                        ].index),
                "positiveLines" :
                        list(self.network.lines[
                                self.network.lines.bus1 == bus
                        ].index),
                "negativeLines" :
                        list(self.network.lines[
                                self.network.lines.bus0 == bus
                        ].index),
                }
        return result


    def getNominalPower(self,generator, time=0,):
        """
        returns the nominal power at a time step saved in the network
        
        Args:
            generator: (str) generator label
            time: (int) index of time slice for which to get nominal power
        Returns:
            (float) maximum power available at generator in time slice at time
        """
        try:
            p_max_pu = self.network.generators_t.p_max_pu[generator].iloc[time]
        except KeyError:
            p_max_pu = 1.0
        return self.network.generators.p_nom[generator] * p_max_pu


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
        builds a dictionary containing all power flows at all time slices for a given
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


    def getLoad(self, bus, time=0, silent=True):
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
            if not silent:
                print(f"Warning: No load at {bus} at timestep {time}.\nFalling back to constant load")
            allLoads = self.network.loads['p_set']
            result = allLoads[allLoads.index.isin(loadsAtCurrentBus)].sum()
        if result < 0:
            raise ValueError(
                "negative Load at current Bus"
            )
        return result

    def getTotalLoad(self, time):
        load = 0.0
        for bus in self.network.buses.index:
            load += self.getLoad(bus,time)
        return load


    def getRepresentingQubits(self, component, time=0):
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


    def getQubitMapping(self,time=0):
        """
        returns a dictionary on which qubits which network components were mapped for
        representation in an ising spin glass problem
        
        Args:
            time: (int) index of time slice for which to get qubit map
        Returns:
            (dict) dictionary with network labels as keys and qubit lists as values
        """
        return {component : self.getRepresentingQubits(component, time)
                for component in self.data.keys() 
                if isinstance(component,str)}
                

    def siquanFormat(self):
        """
        Return the complete problem in the format for the siquan solver
        
        @return: list
            list of tuples of the form (interaction-coefficient, list(qubits))
        """
        return [(v, list(k)) for k, v in self.problem.items() if v != 0 and len(k) > 0]


    def getInteraction(self,*args):
        """
        returns the interaction coeffiecient of a list of qubits
        
        Args:
            *args: (list) a list of integers representing qubits
        Returns:
            (float) the interaction strength between all qubits in args
        """
        sortedUniqueArguments = tuple(sorted(set(args)))
        return self.problem.get(sortedUniqueArguments, 0.0)


    def getHamiltonianMatrix(self,):
        """
        returns a matrix containing the ising hamiltonian
        
        Returns:
            (list) a list of list representing the hamiltonian matrix
        """
        qubits = range(self.allocatedQubits)
        hamiltonian = [
                [self.getInteraction(i,j) for i in qubits] for j in qubits
        ]
        return hamiltonian

    
    def getHamiltonianEigenvalues(self,):
        """
        returns the eigenvalues and normalized eigenvectors of the hamiltonian matrix
        
        Returns:
            (np.ndarray) a numpy array containing all eigenvalues
        """
        return np.linalg.eigh(self.getHamiltonianMatrix())


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


    def calcPowerImbalanceAtBus(self, bus, result, silent=True):
        """
        returns the absolute value of the power imbalance/mismatch at a bus
        
        Args:
            bus: (str) label of the bus at which to calculate power imbalance
            result: (list) list of all qubits which have spin -1 in the solution 

        Returns:
            (dict) dictionary with keys of the type (str, int) over all  time
                        slices and the string alwyays being the chosen bus
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
            if load and not silent:
                print(f"Imbalance at {bus}::{load}")
            contrib[str((bus, t))] = load 
        return contrib

    def calcTotalPowerGeneratedAtBus(self, bus, solution, time=0):
        totalPower = 0.0
        generators = self.getBusComponents(bus)['generators']
        for generator in generators:
            totalPower += self.getEncodedValueOfComponent(generator, solution, time=time)
        return totalPower    

    
    def calcTotalPowerGenerated(self, solution, time=0):
        totalPower = 0.0
        for bus in self.network.buses.index:
            totalPower += self.calcTotalPowerGeneratedAtBus( bus, solution, time=time)
        return totalPower
    

    def calcPowerImbalance(self, solution):
        """
        returns the sum of all absolutes values of power imbalances at each bus.
        This is practically like kirchhoff cost except with linear penalty
        
        Args:
            solution: (list) list of all qubits which have spin -1 in the solution
        Returns:
            (float) the sum of all absolute values of every ower imbalance at every bus
        """
        powerImbalance = 0.0
        for bus in self.network.buses.index:
            for _, imbalance in self.calcPowerImbalanceAtBus(bus, solution).items():
                powerImbalance += abs(imbalance)
        return powerImbalance


    def calcKirchhoffCostAtBus(self, bus, result, silent=True):
        """
        returns a dictionary which contains the kirchhoff cost at the specified bus 'bus' for
        every time slice 'time' as {(bus,time) : value} 

        @param result: list
           
        @return: dict
            dictionary with keys of the type (str,int) over all  time slices and the string 
            alwyays being the chosen bus
        """
        return {
                key : (imbalance * self.kirchhoffFactor) ** 2
                for key, imbalance in self.calcPowerImbalanceAtBus(bus, result, silent=silent).items()
                }


    def calcKirchhoffCost(self, solution):
        """
        calculate the total unscaled kirchhoffcost incurred by a solution
        
        Args:
            solution: (list) list of all qubits which have spin -1 in the solution
            
        Returns:
            (float) total kirchhoff cost incurred without kirchhoffFactor scaling
        """
        kirchhoffCost = 0.0
        for bus in self.network.buses.index:
            for _, val in self.calcPowerImbalanceAtBus(bus, solution).items():
                kirchhoffCost += val ** 2
        return kirchhoffCost


    def individualCostContribution(self, result, silent=True):
        """
        returns a dictionary which contains the kirchhoff cost incurred at every bus at
        every time slice scaled by the KirchhoffFactor

        @param result: list
           list of all qubits which have spin -1 in the solution 
        @return: dict
            dictionary with keys of the form (str,int) over all busses and time slices
        """
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib, **self.calcKirchhoffCostAtBus(bus, result, silent=silent)}
        return contrib


    def individualKirchhoffCost(self, solution, silent=True):
        """
        returns a dictionary which contains the kirchhoff cost incurred at every bus at 
        every time slice without being scaled by the kirchhofffactor
        
        Args:
            solution: (list) list of all qubits which have spin -1 in the solution
        Returns:
            dictionary with keys of the form (str,int) over all busses and time slices
        """
        return {
                key : imbalance ** 2
                for key, imbalance in self.individualPowerImbalance(bus, result, silent=silent).items()
                }


    def individualPowerImbalance(self, solution, silent=True):
        """
        returns a dictionary which contains the power imbalance at each bus at every time slice
        with respect to their type (too much or to little power) via it's sign
        
        Args:
            solution: (list) list of all qubits which have spin -1 in the solution
            silent: (bool) true if the steps when building the result should not print anything
        Returns:
            (dict) dictionary with keys of the form (str, int) over all busses and time slices
        """
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib, **self.calcPowerImbalanceAtBus(bus, solution, silent=silent)}
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
                    marginalCost += self.network.generators["marginal_cost"].loc[generator] * \
                                    self.data[generator]['weights'][0]
            contrib[str((bus, time))] = marginalCost
        return contrib


    def calcMarginalCost(self, solution):
        """
        calculate the total marginal cost incurred by a solution

        @param result: list
            list of all qubits which have spin -1 in the solution
        @return: float
            total marginal cost incurred without monetaryFactor scaling
        """
        marginalCost = 0.0
        for key, val in self.individualMarginalCost(solution).items():
            marginalCost += val 
        return marginalCost


    def calcCost(self, result, ):
        """
        calculates the energy of a spin state including the constant energy contribution
        
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
        *key, interactionStrength = args
        key = tuple(sorted(key))
        for qubit in key:
            interactionStrength *= self.data[qubit]

        # if we couple two spins, we check if they are different. If both spins are the same, 
        # we substitute the product of spins with 1, since 1 * 1 = -1 * -1 = 1 holds. This
        # makes it into a constant contribution. Doesn't work for higer order interactions
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
        componentAdress = self.getRepresentingQubits(component, time)
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
            index of time slice of the first component for which to couple qubits representing it
        @param additionalTime: int
            index of time slice of the second component for which to couple qubits representing it.
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
        firstComponentAdress = self.getRepresentingQubits(firstComponent, time)
        secondComponentAdress = self.getRepresentingQubits(secondComponent, additionalTime)
        # components with 0 weight (power, capacity) vanish in the QUBO formulation
        if (not firstComponentAdress) or (not secondComponentAdress):
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


class fullsplitIsingInterface(IsingPypsaInterface):
    """
    This class provides a line splitting method by using as many integer sized steps
    as possible. A line of capacity 'c' is thus represented by 2*c qubits
    of weight 1.
    """
    def __init__(self, network, snapshots):
        super().__init__(network, snapshots)
        # read generators and lines from network and encode as qubits
        self.storeGenerators()
        self.storeLines()

    def splitCapacity(self, capacity):
        """
        Method to split a line which has maximum capacity "capacity". A line split is a 
        list of lines with varying capacity which can either be on or off. The status of 
        a line is binary, so it can either carry no power or power equal to it's capacity
        The direction of flow for each line is also fixed. It is not enforced that a
        chosen split can only represent flow lower than capacity or all flows that are
        admissable. the line split is represented by the capacity of the it's components

        @param capacity: int
            the capacity of the line that is to be split up
        @return: list
            a list of integers. Each integer is the capacity of a line of the splitting
            direction of flow is encoded as the sign
        """
        return [1 for _ in range(0,capacity,1)]  + [-1 for _ in range(0,capacity,1)]


class fullsplitDirectInefficiencyPenalty(fullsplitIsingInterface):
    def __init__(self, network, snapshots):
        super().__init__(network, snapshots)
        # problem formulation specific parameters
        # problem constraints: kirchhoff, startup/shutdown, marginal cost
        for time in range(len(self.snapshots)):
            for node in self.network.buses.index:
                self.encodeKirchhoffConstraint(node,time)
                self.encodeMarginalCosts(node,time)
                self.encodeStartupShutdownCost(node,time)

    def calcEffiencyLoss(self, cheapGen, expensiveGen,time=0):
        """
        calculates an approximation of the loss of using a generator with higher operational 
        costs compared to using the generator with a lower cost 

        @param cheapGen: str
            generator label with lower marginal cost per MW produced
        @param expensiveGen: str
            generator label that has hower marginal cost per MW produced
        @return: float:
            a value that can be used as a coupling strength for penalizing using the expensive generator
        """
        cheapCost = self.network.generators["marginal_cost"].loc[cheapGen] 
        expensiveCost = self.network.generators["marginal_cost"].loc[expensiveGen] 
        result = max(0,(expensiveCost - cheapCost )) * self.monetaryCostFactor 
        return result


    def encodeMarginalCosts(self, bus, time):
        """
        encodes marginal costs for running generators at a single bus by penalizing expensive
        generators and adding rewards to that generator if cheaper generators are also on
        Doesn't even come close to an optimum

        @param bus: str
            label of the bus at which to add marginal costs
        @param time: int
            index of time slice for which to add marginal costs
        @return: None
            modifies self.problem. Adds to previously written interaction cofficient
        """
        FACTOR = 1.0
        generators = self.getBusComponents(bus)['generators']
        sortedGenerators = sorted(
                generators,
                key= lambda gen : self.network.generators["marginal_cost"].loc[gen]
        )
        numGenerators = len(generators)
        for idx, cheapGen in enumerate(sortedGenerators):
            for _, expensiveGen in enumerate(sortedGenerators, start=idx + 1):
                couplingStrength = self.calcEffiencyLoss(cheapGen, expensiveGen)
                self.coupleComponentWithConstant(
                    expensiveGen,
                    couplingStrength =   1.0 / (len(generators) - idx) * \
                            self.getNominalPower(cheapGen, time) * \
                            couplingStrength * \
                            FACTOR,
                    time=time
                ) 
                self.coupleComponents(
                    cheapGen,
                    expensiveGen,
                    couplingStrength=  - 1.0 / (len(generators)- idx) * \
                            couplingStrength * \
                            FACTOR,
                    time=time
                )
        return


class fullsplitNoMarginalCost(fullsplitIsingInterface):
    """
    This class uses a 'fullsplit' to encode lines. It optimizes according to the kirchhoff
    constraint, but doesn't consider any marginal costs incurred
    """
    def __init__(self, network, snapshots):
        super().__init__(network, snapshots)
        # problem constraints: kirchhoff
        for time in range(len(self.snapshots)):
            for node in self.network.buses.index:
                self.encodeKirchhoffConstraint(node, time)


class fullsplitLocalMarginalEstimationDistance(fullsplitIsingInterface):
    """
    class for building an ising spin glass problem for optimizing marginal costs
    while respecting the kirchoff constraint. We represent it by substituting generator
    marginal cost by their difference to the most efficient generator. then we estimate
    a lower bound of the marginal cost at every bus. The marginal cost constraint is then given
    as minimizing the squared distance of incurred marginal cost to the estimated marginal cost
    This method doesn't pay attention to line transmission changing where marginal costs
    are incurred
    """
    def __init__(self, network, snapshots):
        super().__init__(network, snapshots)
        # problem constraints: kirchhoff, startup/shutdown, marginal cost
        envMgr = EnvironmentVariableManager()
        # factor to scale the offset of marginal cost when estimating
        # marginal cost at a bus
        self.offsetEstimationFactor = float(envMgr["offsetEstimationFactor"])
        # factor to scale estimated cost at a bus after calculation
        self.estimatedCostFactor = float(envMgr["estimatedCostFactor"])
        # factor to scale marginal cost of a generator when constructing ising
        # interactions
        self.offsetBuildFactor = float(envMgr["offsetBuildFactor"])
        for time in range(len(self.snapshots)):
            for node in self.network.buses.index:
                self.encodeKirchhoffConstraint(node,time)
                self.encodeMarginalCosts(node,time)
                self.encodeStartupShutdownCost(node,time)


    def chooseOffset(self, sortedGenerators):
        """
        calculates a float by which to offset all marginal costs. The chosen offset is the
        minimal marginal cost of a generator in the list

        @param sortedGenerators: list
            a list of generators already sorted by their minimal cost in ascending order
        @return: float
             a float by which to offset all marginal costs of network components
        """
        # there are lots of ways to choose an offset. offsetting such that 0 is minimal cost
        # is decent but for example choosing an offset slighty over that seems to also produce
        # good results. It is not clear how important the same sign on all marginal costs is
        marginalCostList = [self.network.generators["marginal_cost"].loc[gen] for gen in sortedGenerators]
        return self.offsetEstimationFactor * np.min(marginalCostList) 


    def estimateMarginalCostAtBus(self, bus, time):
        """
        estimates a lower bound for marginal costs incurred by matching the load at the bus
        only with generators that are at this bus
        
        @param bus: str
            but solution which to estimate marginal costs
        @param time: int
            index of time slice for which to estimate marginal costs
        @return: float, float
            returns an estimation of the incurred marginal cost if the marginal costs of generators
            are all offset by the second return value
        """
        remainingLoad = self.getLoad(bus, time) 
        generators = self.getBusComponents(bus)['generators']
        sortedGenerators = sorted(
                generators,
                key= lambda gen : self.network.generators["marginal_cost"].loc[gen]
        )
        offset = self.chooseOffset(sortedGenerators)
        costEstimation = 0.0
        for generator in sortedGenerators:
            if remainingLoad <= 0:
                break
            suppliedPower = min(remainingLoad, self.data[generator]['weights'][0])
            costEstimation += suppliedPower * (self.network.generators["marginal_cost"].loc[generator] - offset)
            remainingLoad -= suppliedPower
        return costEstimation, offset

    def calculateCost(self, componentToBeValued, allComponents, offset, estimatedCost, load, bus):
        if componentToBeValued in allComponents['generators']:
            return self.network.generators["marginal_cost"].loc[componentToBeValued] - offset
        if componentToBeValued in allComponents['positiveLines']:
            return   0.5 *   estimatedCost / load
        if componentToBeValued in allComponents['negativeLines']:
            return  0.5 * - estimatedCost / load


    def encodeMarginalCosts(self, bus, time):
        """
        encodes marginal costs at a bus by first estimating a lower bound of unavoidable marginal costs
        Then deviation in the marginal cost from that estimation are penalized quadratically
        
        @param bus: str
            but at which to estimate marginal costs
        @param time: int
            index of time slice for which to estimate marginal costs
        @return: None 
             modifies self.problem. Adds to previously written interaction cofficient 
        """
        components = self.getBusComponents(bus)
        flattenedComponenents = components['generators'] + \
                components['positiveLines'] + \
                components['negativeLines']

        estimatedCost, offset = self.estimateMarginalCostAtBus(bus,time)
        estimatedCost *= self.estimatedCostFactor
        offset *= self.offsetBuildFactor
        load = self.getLoad(bus, time)

        self.addInteraction(0.25 * estimatedCost ** 2)
        for firstComponent in flattenedComponenents:
            self.coupleComponentWithConstant(
                    firstComponent,
                    - 2.0 * self.calculateCost(firstComponent, components, offset, estimatedCost, load, bus) * \
                            estimatedCost *  \
                            self.monetaryCostFactor 
                    )
            for secondComponent in flattenedComponenents:
                curFactor = self.monetaryCostFactor * \
                                self.calculateCost(firstComponent, components, offset, estimatedCost, load, bus) * \
                                self.calculateCost(secondComponent, components, offset, estimatedCost, load, bus) 
                self.coupleComponents(
                        firstComponent,
                        secondComponent,
                        couplingStrength=curFactor
                )


class fullsplitMarginalAsPenalty(fullsplitIsingInterface):

    def __init__(self, network, snapshots):
        super().__init__(network, snapshots)
        # factor to scale the offset of marginal cost 
        envMgr = EnvironmentVariableManager()
        self.offsetEstimationFactor = float(envMgr["offsetEstimationFactor"])
        # problem constraints: kirchhoff, startup/shutdown, marginal cost
        for time in range(len(self.snapshots)):
            for node in self.network.buses.index:
                self.encodeKirchhoffConstraint(node,time)
                self.encodeMarginalCosts(node,time)
                self.encodeStartupShutdownCost(node,time)

    def marginalCostOffset(self, ):
        """
        returns a float by which all generator marginal costs per power will be offset.
        Since every generator will be offset, this will not change relative costs between them
        It changes the range of energy contributions this constraint provides. Adding marginal
        costs as a cost to the QUBO formulation will penalize all generator configurations. The offset
        shifts it so that the cheapest generator doesn't get any penalty
        
        @return: float
            a float that in is in the range of generator marginal costs
        """
        return 1.0 * min(self.network.generators["marginal_cost"]) * self.offsetEstimationFactor

    def encodeMarginalCosts(self, bus, time):
        """
        encodes marginal costs for running generators and transmission lines at a single bus.
        This uses an offset calculated in ´marginalCostOffset´, which is a dependent on all generators
        of the entire network for a single time slice

        @param bus: str
            label of the bus at which to add marginal costs
        @param time: int
            index of time slice for which to add marginal cost
        @return: None
            modifies self.problem. Adds to previously written interaction cofficient 
        """
        generators = self.getBusComponents(bus)['generators']
        costOffset = self.marginalCostOffset()
        for generator in generators:
            self.coupleComponentWithConstant(
                    generator, 
                    couplingStrength=self.monetaryCostFactor * \
                            (self.network.generators["marginal_cost"].loc[generator] - \
                            costOffset),
                    time=time
            )


class fullsplitMarginalAsPenaltyAverageOffset(fullsplitMarginalAsPenalty):
    """
    An extension of the fullsplitMarginalAsPenalty strategy by replacing
    the cost offset equal to the minimal marginal cost with a slightly below
    average of energy cost
    """
    def calcAverageCostPerPowerGenerated(self, time=0):
        """
        calculates average cost power unit produced if all generators
        were switched on at the first time slice

        @param time: int
            index of time slice for which to calculate average marginal costs
        @return: float
            the average marginal cost of all generators weighted by power output
        """
        maxCost = 0.0
        maxPower = 0.0
        for generator in self.network.generators.index:
            currentPower = self.getNominalPower(generator, time)
            maxCost += currentPower * self.network.generators["marginal_cost"].loc[generator]
            maxPower += currentPower
        return float(maxCost / maxPower)

    def marginalCostOffset(self, time=0):
        """
        returns a float by which all generator marginal costs per power will be offset.
        Since every generator will be offset, this will not change relative costs between them
        It changes the range of energy contributions this constraint provides. Adding marginal
        costs as a cost to the QUBO formulation will penalize all generator configurations. The offset
        shifts it by 90% of the total average of energy cost such that the added total energy cost
        is close enough to 0
        
        @return: float
            a float that in is in the range of generator marginal costs
        """
        return 1.0 * self.calcAverageCostPerPowerGenerated(time)


class fullsplitGlobalCostSquare(fullsplitIsingInterface):
    """
    class for building an ising spin glass problem for optimizing marginal costs
    while respecting the kirchhoff constraint. The marginal costs of using generators
    are considered one single global constraint. The square of marginal costs is encoded
    into the energy and thus minimized
    """
    def __init__(self, network, snapshots):
        # problem constraints: kirchhoff, startup/shutdown, marginal cost
        envMgr = EnvironmentVariableManager()
        # factor to scale the offset of marginal cost when estimating
        # marginal cost at a bus
        self.offsetEstimationFactor = float(envMgr["offsetEstimationFactor"])
        # factor to scale estimated cost at a bus after calculation
        self.estimatedCostFactor = float(envMgr["estimatedCostFactor"])
        # factor to scale marginal cost of a generator when constructing ising
        # interactions
        self.offsetBuildFactor = float(envMgr["offsetBuildFactor"])
        super().__init__(network, snapshots)
        for time in range(len(self.snapshots)):
            self.encodeMarginalCosts(time)
            for node in self.network.buses.index:
                self.encodeKirchhoffConstraint(node,time)
                self.encodeStartupShutdownCost(node,time)


        gen = self.network.generators.index
        demand = self.getTotalLoad(0)
        scale = 0.0
        print(demand)

        # constant load contribution to cost function so that a configuration that fulfills the
        # kirchhoff contraint has energy 0
        self.addInteraction((scale * demand) ** 2)
        for component1 in gen:
            factor = 1.0
            # reward/penalty term for matching/adding load
            self.coupleComponentWithConstant(component1, - 2.0 * scale * factor * demand)
            for component2 in gen:
                curFactor = factor * scale
                # attraction/repulsion term for different/same sign of power at components
                self.coupleComponents(component1, component2, couplingStrength=curFactor)


    def chooseOffset(self, sortedGenerators):
        """
        calculates a float by which to offset all marginal costs. The chosen offset is the
        minimal marginal cost of a generator in the list

        @param sortedGenerators: list
            a list of generators already sorted by their minimal cost in ascending order
        @return: float
             a float by which to offset all marginal costs of network components
        """
        # there are lots of ways to choose an offset. offsetting such that 0 is minimal cost
        # is decent but for example choosing an offset slighty over that seems to also produce
        # good results. It is not clear how important the same sign on all marginal costs is
        marginalCostList = [self.network.generators["marginal_cost"].loc[gen] for gen in sortedGenerators]
        return self.offsetEstimationFactor * np.min(marginalCostList) 


    def estimateGlobalMarginalCost(self, time, expectedAdditonalCost=0.0):
        """
        estimates a lower bound of incurred marginal costs if locality of generators could be
        ignored at a given time slice. Unavoidable baseline costs of matching the load is ignored. 
        The offset to reduce baseline costs to 0 and estimated marginal cost with a constant is returned

        @param time: int
            index of time slice for which to calculate lower bound of offset marginal cost
        @param expectedAdditonalCost: float
            constant by which to offset the returned marginal cost bound
        @return: float, float
            lower bound of global marginal cost
            offset that was subtracted from marginal costs of all generators to calculate lower bound
        """
        load = 0.0
        for bus in self.network.buses.index:
            load += self.getLoad(bus,time)

        sortedGenerators = sorted(
                self.network.generators.index,
                key= lambda gen : self.network.generators["marginal_cost"].loc[gen]
        )
        offset = self.chooseOffset(sortedGenerators)
        costEstimation = 0.0
        for generator in sortedGenerators:
            if load <= 0:
                break
            suppliedPower = min(load, self.data[generator]['weights'][0])
            costEstimation += suppliedPower * (self.network.generators["marginal_cost"].loc[generator] - offset)
            load -= suppliedPower
        return costEstimation+expectedAdditonalCost, offset


    def encodeMarginalCosts(self, time):
        """
        The marginal costs of using generators
        are considered one single global constraint. The square of marginal costs is encoded
        into the energy and thus minimized

        @param time: int
            index of time slice for which to estimate marginal costs
        @param expectedAdditonalCost: float
            float by which lower estimate off marginal cost is offset
        @return: None 
            modifies self.problem. Adds to previously written interaction cofficient 
        """
        estimatedCost , offset = self.estimateGlobalMarginalCost(time,expectedAdditonalCost= 0)
        generators = self.network.generators.index

        load = 0.0
        for bus in self.network.buses.index:
            load += self.getLoad(bus,time)

        print(f"Offset: {offset}")
        print(f"Minimal estimated Cost (with offset): {estimatedCost}")
        print(f"Load: {load}")
        print(f"Current total estimation at {time}: {offset * self.getTotalLoad(time)}")
        for gen1 in generators:
            marginalCostGen1 = self.network.generators["marginal_cost"].loc[gen1] - offset
            for gen2 in generators:
                marginalCostGen2 = self.network.generators["marginal_cost"].loc[gen2] - offset
                curFactor = self.monetaryCostFactor * \
                                marginalCostGen1 * \
                                marginalCostGen2 
                self.coupleComponents(
                        gen1,
                        gen2,
                        couplingStrength=curFactor
                )


class binarysplitIsingInterface(IsingPypsaInterface):
    """
    This class provides a line splitting method by using two qubits for full power flow in
    either direction
    """
    def __init__(self, network, snapshots):
        super().__init__(network, snapshots)
        # read generators and lines from network and encode as qubits
        self.storeGenerators()
        self.storeLines()

    def splitCapacity(self, capacity):
        """
        Method to split a line which has maximum capacity "capacity". A line split is a 
        list of lines with varying capacity which can either be on or off. The status of 
        a line is binary, so it can either carry no power or power equal to it's capacity
        The direction of flow for each line is also fixed. It is not enforced that a
        chosen split can only represent flow lower than capacity or all flows that are
        admissable. the line split is represented by the capacity of the it's components

        @param capacity: int
            the capacity of the line that is to be split up
        @return: list
            a list of integers. Each integer is the capacity of a line of the splitting
            direction of flow is encoded as the sign
        """
        return [capacity, - capacity]


class binarysplitNoMarginalCost(binarysplitIsingInterface):
    """
    This class uses a 'binarysplit' to encode lines. It optimizes according to the kirchhoff
    constraint, but doesn't consider any marginal costs incurred
    """
    def __init__(self, network, snapshots):
        super().__init__(network, snapshots)
        # problem constraints: kirchhoff
        for time in range(len(self.snapshots)):
            for node in self.network.buses.index:
                self.encodeKirchhoffConstraint(node,time)
