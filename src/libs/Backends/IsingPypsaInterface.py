from collections import OrderedDict
import numpy as np
import typing

import pypsa


def fullsplit(capacity: int) -> list:
    """
    A method for splitting up the capacity of a line with a given
    maximum capacity.
    A line is split into qubits with weights just that any sum of
    acitve qubits never exceeds the capacity and so that it can
    represent any value that is smaller than the capacity. This method
    archives this by splitting the lines into qubits which all have
    either weight 1 or -1.
    
    Args:
        capacity: (int)
            The capacity of the line to be decomposed.
    Returns:
        (list)
            A list of weights to be used in decomposing a line.
    """
    return [1] * capacity + [-1] * capacity


def binarysplit(capacity: int) -> list:
    """
    A method for splitting up the capacity of a line with a given
    maximum capacity.
    A line is split into qubits with weights of the given capacity, or
    its inverse. This way the line can carry energy bidirectional.

    Args:
        capacity: (int)
            The capacity of the line to be decomposed.
    Returns:
        (list)
            A list of weights to be used in decomposing a line.
    """
    return [capacity, -capacity]


def customsplit(capacity: int) -> list:
    """
    A method for splitting up the capacity of a line with a given
    maximum capacity.
    A line is split into qubits with weights, which are determined in a
    custom manner, where the sum of positive and the sum of negative
    weights is always the given capacity.

    Args:
        capacity: (int)
            The capacity of the line to be decomposed.
    Returns:
        (list)
            A list of weights to be used in decomposing a line.
    """
    if capacity == 0:
        return []
    if capacity == 1:
        return [1, -1]
    if capacity == 2:
        return [2, -1, -1]
    if capacity == 3:
        return [3, -2, -1]
    if capacity == 4:
        return [2, 2, -3, -1, ]
    if capacity == 5:
        return [4, 1, -3, -2]
    raise ValueError("Capacity is too big to be decomposed")


def binaryPower(number: int) -> list:
    """
    return a cut off binary representation of the argument. It is a list
    of powers of two and a rest such that the sum over the list is equal
    to the number.
    
    Args:
        number: (int)
            The integer to be decomposed
    Returns:
        (list)
            List of integer whose sum is equal to the number
    """
    if number < 0:
        raise ValueError
    if number == 0:
        return []
    number_of_bits = number.bit_length()
    result = [2 ** exp for exp in range(number_of_bits - 1)]
    return result + [number - 2 ** (number_of_bits - 1) + 1]


class IsingBackbone:
    """
    This class implements the conversion of a unit commitment problem
    given by a Pypsa network to an Ising spin glass problem.
    It acts as an endpoint to decode qubit configuration and encode
    coupling of network components into Ising interactions. It encodes
    the various network components into qubits and provides methods to
    interact with those qubits based on the label of the network
    component they represent. Therefore, it only acts as a data
    structure which other objects can use to model a specific
    problem/constraint. Modeling of various constraints is delegated
    to instances of `IsingSubproblem`, which are stored as attributes.
    An IsingSubproblem provides a method which adds the Ising
    representation of the subproblem it models to the stored Ising
    problem in this class.
    Extending the Ising model of the network can be done in two ways. If
    you want to extend which kinds of networks can be read, you have to
    extend this class with methods that convert values of the network
    into qubits and write appropriate access methods. If you want to
    add a new constraint, you have to write a class that adheres to the
    `AbstractIsingSubproblem` interface
    """
    # dictionary that maps config strings to the corresponding classes and
    # methods
    linesplitDict = {
        "fullsplit": fullsplit,
        "binarysplit": binarysplit,
        "customsplit": customsplit,
    }

    def __init__(self, network: pypsa.Network, linesplitName: str,
                 configuration: dict):
        """
        Constructor for an Ising Backbone. It requires a network and
        the name of the function that defines how to encode lines. Then
        it goes through the configuration dictionary and encodes all
        sub-problems present into the instance.

        Args:
            network: (pypsa.Network)
                The pypsa network which to encode into qubits.
            linesplitName: (str)
                The name of the linesplit function as given in
                linesplitDict.
            configuration: (dict)
                A dictionary containing all subproblems to be encoded
                into an ising problem.
        """
        self.subproblemTable = {
            "kirchhoff": KirchhoffSubproblem,
            "marginalCost": MarginalCostSubproblem
        }
        if "kirchhoff" not in configuration:
            print("No Kirchhoff configuration found, "
                  "adding Kirchhoff constraint with Factor 1.0")
            configuration["kirchhoff"] = {"scaleFactor": 1.0}

        # resolve string for splitting line capacty to function
        self._linesplitName = linesplitName
        self.splitCapacity = IsingBackbone.linesplitDict[linesplitName]

        # network to be solved
        self.network = network
        self.snapshots = network.snapshots

        # contains ising coefficients
        self.problem = {}
        # mirrors encodings of `self.problem`, but is reset after encoding a
        # subproblem to get ising formulations of subproblems
        self.cachedProblem = {}

        # initializing data structures that encode the network into qubits
        self.data = {}
        self.allocatedQubits = 0
        self.storeGenerators()
        self.storeLines()

        # read configuration dict, store in _subproblems and apply encodings
        self._subproblems = {}
        # dicitionary of all support subproblems
        for subproblem, subproblemConfiguration in configuration.items():
            if subproblem not in self.subproblemTable:
                print(f"{subproblem} is not a valid subproblem, skipping "
                      f"encoding")
                continue
            if not subproblemConfiguration:
                print(f"Subproblem {subproblem} has no configuration data, "
                      f"skipping encoding")
                continue
            subproblemInstance = self.subproblemTable[
                subproblem].buildSubproblem(self, subproblemConfiguration)
            self._subproblems[subproblem] = subproblemInstance
            self.flushCachedProblem()
            subproblemInstance.encodeSubproblem(self)

    def __getattr__(self, method_name: str) -> callable:
        """
        This function delegates method calls to an IsingBackbone to a
        subproblem instance if IsingBackbone doesn't have such a method.
        We can use this by calling subproblem methods from the backbone
        instance. If the name of the method is not unique among all
        subproblems it will raise an attribute error.

        Args:
            method_name: (str)
                The name of the method to be delegated to a subproblem.
        Returns:
            (callable)
                The corresponding method of an ising subproblem.
        """
        method = None
        uniqueResolution = True
        for subproblem, subproblemInstance in self._subproblems.items():
            if hasattr(subproblemInstance, method_name):
                if uniqueResolution:
                    uniqueResolution = False
                    method = getattr(subproblemInstance, method_name)
                else:
                    raise AttributeError(f"{method_name} didn't resolve to "
                                         f"unique subproblem")
        if method:
            return method
        else:
            raise AttributeError(f"{method_name} was not found in any stored "
                                 f"subproblem")

    # obtain config file using an reader
    @classmethod
    def buildIsingProblem(cls, network: pypsa.Network, config: dict):
        """
        This is a factory method for making an IsingBackbone that
        corresponds to the problem in the network.
        First, it retrieves information from the config dictionary on
        which kind of IsingBackbone to build and then returns the
        IsingBackbone object made using the rest of the configuration.

        Args:
            network: (pypsa.Network)
                The pypsa network problem to be cast into QUBO/Ising
                form.
            config: (dict)
                A dictionary containing all information about the QUBO
                formulations.
        Returns:
            (IsingBackbone)
                An IsingBackbone that models the unit commitment problem
                of the network.
        """
        linesplitFunction = config.pop("formulation")
        return IsingBackbone(network, linesplitFunction, config)

    def flushCachedProblem(self) -> None:
        """
        Resets the cached changes of interactions.

        Returns:
            (None)
        """
        self.cachedProblem = {}

    # functions to couple components. The couplings are interpreted as
    # multiplications of QUBO polynomials. The final interactions are
    # coefficients for an ising spin glass problem
    def addInteraction(self, *args) -> None:
        """
        This method is used for adding a new Ising interactions between
        multiple qubits to the problem.
        The interaction is scaled by all qubits specific weights. For
        higher order interactions, it performs substitutions of qubits
        that occur multiple times, which would be constant in an Ising
        spin glass problem. Interactions are stored in the attribute
        `problem`, which is a dictionary. Keys of that dictionary are
        tuples of involved qubits and a value, which is a float.
        The method can take an arbitrary number of arguments, where the
        last argument is the interaction strength. All previous
        arguments are assumed to contain spin ids.

        Args:
            args[-1]: (float)
                The basic interaction strength before applying qubit
                weights.
            args[:-1]: (int)
                All qubits that are involved in this interaction.
        Returns:
            (None)
                Modifies self.problem by adding the strength of the
                interaction if an interaction coefficient is already
                set.
        """
        if len(args) > 3:
            raise ValueError(
                "Too many arguments for an interaction"
            )
        *key, interactionStrength = args
        key = tuple(sorted(key))
        for qubit in key:
            interactionStrength *= self.data[qubit]

        # if we couple two spins, we check if they are different. If both spins
        # are the same, we substitute the product of spins with 1, since
        # 1 * 1 = -1 * -1 = 1 holds. This makes it into a constant
        # contribution. Doesn't work for higher order interactions
        if len(key) == 2:
            if key[0] == key[1]:
                key = tuple([])
        self.problem[key] = self.problem.get(key, 0) - interactionStrength
        self.cachedProblem[key] = self.cachedProblem.get(key, 0) \
                                  - interactionStrength

    # TODO unify coupleComponents and with Constant
    def coupleComponentWithConstant(self, component: str,
                                    couplingStrength: float = 1,
                                    time: int = 0) -> None:
        """
        Performs a QUBO multiplication involving a single variable on
        all qubits which are logically grouped to represent a component
        at a given time slice. This QUBO multiplication is translated
        into Ising interactions and then added to the currently stored
        Ising spin glass problem.

        Args:
            component: (str)
                Label of the network component.
            couplingStrength: (float)
                Cofficient of QUBO multiplication by which to scale the
                interaction. Does not contain qubit specific weight.
            time: (int)
                Index of time slice for which to couple qubit
                representing the component.

        Returns:
            (None)
                Modifies self.problem. Adds to previously written
                interaction coefficient.
        """
        componentAdress = self.getRepresentingQubits(component, time)
        for qubit in componentAdress:
            # term with single spin after applying QUBO to Ising transformation
            self.addInteraction(qubit, 0.5 * couplingStrength)
            # term with constant cost constribution after applying QUBO to
            # Ising transformation
            self.addInteraction(0.5 * couplingStrength * self.data[qubit])

    # TODO add a method to conveniently encode the squared distance to a fixed
    #  value into an ising

    def coupleComponents(self,
                         firstComponent: str,
                         secondComponent: str,
                         couplingStrength: float = 1,
                         time: int = 0,
                         additionalTime: int = None
                         ) -> None:
        """
        This method couples two labeled groups of qubits as a product
        according to their weight and the selected time step.
        It performs a QUBO multiplication involving exactly two
        components on all qubits which are logically grouped to
        represent these components at a given time slice. This QUBO
        multiplication is translated into Ising interactions, scaled by
        the couplingStrength and the respective weights of the qubits
        and then added to the currently stored Ising spin glass problem.

        Args:
            firstComponent: (str)
                Label of the first network component.
            secondComponent: (str)
                Label of the second network component.
            couplingStrength: (float)
                Coefficient of QUBO multiplication by which to scale all
                interactions.
            time: (int)
                Index of time slice of the first component for which to
                couple qubits representing it.
            additionalTime: (int)
                Index of time slice of the second component for which
                to couple qubits representing it. The default parameter
                'None' is used if the time slices of both components
                are the same.
        Returns:
            (None)
                Modifies `self.problem`. Adds to previously written
                interaction coefficient.

        Example:
            Let X_1, X_2 be the qubits representing firstComponent and
            Y_1, Y_2 the qubits representing secondComponent. The QUBO
            product the method translates into Ising spin glass
            coefficients is:
            (X_1 + X_2) * (Y_1 + Y_2) = X_1 * Y_1 + X_1 * Y_2
                                        + X_2 * Y_1 + X_2 * Y_2
        """
        # Replace None default values with their intended network component and
        # then figure out which qubits we want to couple based on the
        # component name and chosen time step
        if additionalTime is None:
            additionalTime = time
        firstComponentAdress = self.getRepresentingQubits(firstComponent, time)
        secondComponentAdress = self.getRepresentingQubits(secondComponent,
                                                           additionalTime)
        # components with 0 weight (power, capacity) vanish in the QUBO
        # formulation
        if (not firstComponentAdress) or (not secondComponentAdress):
            return
        # retrieving corresponding qubits is done. Now perform qubo
        # multiplication by expanding the product and add each summand
        # invididually.
        for first in range(len(firstComponentAdress)):
            for second in range(len(secondComponentAdress)):
                # The body of this loop corresponds to the multiplication of
                # two QUBO variables. According to the QUBO - Ising
                # translation rule x = (sigma+1)/2 one QUBO multiplication
                # results in 4 ising interactions, including constants

                # term with two spins after applying QUBO to Ising
                # transformation if both spin id's are the same, this will
                # add a constant cost.
                # addInteraction performs substitution of spin with a constant
                self.addInteraction(
                    firstComponentAdress[first],
                    secondComponentAdress[second],
                    couplingStrength * 0.25
                )
                # terms with single spins after applying QUBO to Ising
                # transformation
                self.addInteraction(
                    firstComponentAdress[first],
                    couplingStrength * self.data[secondComponent]['weights'][
                        second] * 0.25
                )
                self.addInteraction(
                    secondComponentAdress[second],
                    couplingStrength * self.data[firstComponent]['weights'][
                        first] * 0.25
                )
                # term with constant cost constribution after applying QUBO to
                # Ising transformation
                self.addInteraction(
                    self.data[firstComponent]['weights'][first]
                    * self.data[secondComponent]['weights'][second]
                    * couplingStrength * 0.25
                )

    # end of coupling functions

    def numVariables(self) -> int:
        """
        Returns how many qubits have already been used to model the
        problem components.
        When allocating qubits for a new component, those qubits will
        start at the value returned by this method and later updated.

        Returns:
            (int)
                The number of qubits already allocated.
        """
        return self.allocatedQubits

    # create qubits for generators and lines
    def storeGenerators(self) -> None:
        """
        Assigns qubits to each generator in self.network. For each
        generator it writes generator specific parameters (i.e. power,
        corresponding qubits, size of encoding) into the dictionary
        self.data. At last it updates object specific parameters.

        Returns:
            (None)
                Modifies self.data and self.allocatedQubits
        """
        timesteps = len(self.snapshots)
        for generator in self.network.generators.index:
            self.createQubitEntriesForComponent(
                componentName=generator,
                numQubits=timesteps,
                weights=[self.getNominalPower(generator, time) for time in
                         range(len(self.snapshots))],
                encodingLength=1,
            )
        return

    def storeLines(self) -> None:
        """
        Assigns a number of qubits, according to the option set in
        self.config, to each line in self.network. For each line, line
        specific parameters (i.e. power, corresponding qubits, size of
        encoding) are as well written into the dictionary self.data. At
        last it updates object specific parameters.
        
        Returns:
            (None)
                Modifies self.data and self.allocatedQubits
        """
        timesteps = len(self.snapshots)
        for line in self.network.lines.index:
            singleTimestepSplit = self.splitCapacity(
                int(self.network.lines.loc[line].s_nom))
            self.createQubitEntriesForComponent(
                componentName=line,
                weights=singleTimestepSplit * timesteps,
                encodingLength=len(singleTimestepSplit),
            )

    def createQubitEntriesForComponent(self,
                                       componentName: str,
                                       numQubits: int = None,
                                       weights: int = None,
                                       encodingLength: int = None
                                       ) -> None:
        """
        A function to create qubits in the self.data dictionary that
        represent some network components. The qubits can be accessed
        using the componentName.
        The method places several restriction on what it accepts in
        order to generate a valid QUBO later on. The checks are intended
        to prevent name or qubit collision.

        Args:
            componentName: (str)
                The string used to couple the component with qubits.
            numQubits: (int)
                Number of qubits necessary to encode the component.
            weights: (int)
                Weight for each qubit which to use whenever it gets
                coupled with other qubits.
            encodingLength: (int)
                Number of qubits used for encoding during one time step

        Returns:
            (None)
                Modifies self.data and self.allocatedQubits
        """
        if isinstance(componentName, int):
            raise ValueError("Component names mustn't be of type int")
        if componentName in self.data:
            raise ValueError("Component name has already been used")
        if weights is None:
            raise ValueError("Assigned qubits don't have any weight")
        if numQubits is None:
            numQubits = len(weights) * len(self.snapshots)
        if numQubits * len(self.snapshots) != len(weights):
            raise ValueError("Assigned qubits don't match number of weights")
        if len(self.snapshots) * encodingLength != numQubits:
            raise ValueError("total number of qubits, numer of snapshots and "
                             "qubits per snapshot is not consistent")
        indices = range(self.allocatedQubits, self.allocatedQubits + numQubits)
        self.allocatedQubits += numQubits
        self.data[componentName] = {
            'indices': indices,
            'weights': weights,
            'encodingLength': encodingLength,
        }
        for idx, qubit in enumerate(indices):
            self.data[qubit] = weights[idx]

    # helper functions to set encoded values
    def setOutputNetwork(self, solution: list) -> pypsa.Network:
        """
        Writes the status, p and p_max_pu values of generators, and the
        p0 and p1 values of lines according to the provided solution in
        a copy of self.network. This copy is then returned.

        Args:
            solution: (list)
                List of all qubits which have spin -1 in the solution.

        Returns:
            (pypsa.Network)
                A copy of self.network in which time-dependant values
                for the generators and lines are set according to the
                given solution.
        """
        outputNetwork = self.network.copy()
        # get Generator/Line Status
        lines = {}
        for time in range(len(self.snapshots)):
            for generator in outputNetwork.generators.index:
                # set value in status-dataframe in generators_t dictionary
                status = int(
                    self.getGeneratorStatus(gen=generator,
                                            solution=solution,
                                            time=time))
                column_status = list(outputNetwork.generators_t.status.columns)
                if generator in column_status:
                    index_generator = column_status.index(generator)
                    outputNetwork.generators_t.status.iloc[
                        time, index_generator] = status
                else:
                    outputNetwork.generators_t.status[generator] = status

                # set value in p-dataframe in generators_t dictionary
                p = self.getEncodedValueOfComponent(component=generator,
                                                    solution=solution,
                                                    time=time)
                columns_p = list(outputNetwork.generators_t.p.columns)
                if generator in columns_p:
                    index_generator = columns_p.index(generator)
                    outputNetwork.generators_t.p.iloc[
                        time, index_generator] = p
                else:
                    outputNetwork.generators_t.p[generator] = p

                # set value in p_max_pu-dataframe in generators_t dictionary
                columns_p_max_pu = list(
                    outputNetwork.generators_t.p_max_pu.columns)
                p_nom = outputNetwork.generators.loc[generator, "p_nom"]
                if p == 0:
                    p_max_pu = 0.0
                else:
                    p_max_pu = p_nom / p
                if generator in columns_p_max_pu:
                    index_generator = columns_p_max_pu.index(generator)
                    outputNetwork.generators_t.p_max_pu.iloc[
                        time, index_generator] = p_max_pu
                else:
                    outputNetwork.generators_t.p_max_pu[generator] = p_max_pu

            for line in outputNetwork.lines.index:
                encoded_val = self.getEncodedValueOfComponent(
                    component=line,
                    solution=solution,
                    time=time)
                # p0 - Active power at bus0 (positive if branch is withdrawing
                # power from bus0).
                # p1 - Active power at bus1 (positive if branch is withdrawing
                # power from bus1).
                p0 = encoded_val
                p1 = -encoded_val

                columns_p0 = list(outputNetwork.lines_t.p0.columns)
                if line in columns_p0:
                    index_line = columns_p0.index(line)
                    outputNetwork.lines_t.p0.iloc[time, index_line] = p0
                else:
                    outputNetwork.lines_t.p0[line] = p0

                columns_p1 = list(outputNetwork.lines_t.p1.columns)
                if line in columns_p1:
                    index_line = columns_p1.index(line)
                    outputNetwork.lines_t.p1.iloc[time, index_line] = p1
                else:
                    outputNetwork.lines_t.p1[line] = p1

        return outputNetwork

    # helper functions for getting encoded values
    def getData(self) -> dict:
        """
        Returns the dictionary that holds information on the encoding
        of the network into qubits.
        
        Returns:
            (dict)
                The dictionary with network component as keys and qubit
                information as values
        """
        return self.data

    def getBusComponents(self, bus: str) -> dict:
        """
        Returns all labels of components that connect to a bus as a
        dictionary. For lines that end in this bus, positive power flow
        is interpreted as increasing available power at the bus. For
        lines that start in this bus positive power flow is interpreted
        as decreasing available power at the bus.

        Args:
            bus: (str)
                Label of the bus.

        Returns:
            (dict)
                A dictionary with the three following keys:
                'generators':       A list of labels of generators that
                                    are at the bus.
                'positiveLines':    A list of labels of lines that end
                                    in this bus.
                'negativeLines':    A list of labels of lines that start
                                    in this bus.
        """
        if bus not in self.network.buses.index:
            raise ValueError("the bus " + bus + " doesn't exist")
        result = {
            "generators":
                list(self.network.generators[
                         self.network.generators.bus == bus
                         ].index),
            "positiveLines":
                list(self.network.lines[
                         self.network.lines.bus1 == bus
                         ].index),
            "negativeLines":
                list(self.network.lines[
                         self.network.lines.bus0 == bus
                         ].index),
        }
        return result

    def getNominalPower(self, generator: str, time: int = 0) -> float:
        """
        Returns the nominal power of a generator at a time step saved
        in the network.
        
        Args:
            generator: (str)
                The generator label.
            time: (int)
                Index of time slice for which to get nominal power

        Returns:
            (float)
                Nominal power available at 'generator' at time slice
                'time'
        """
        try:
            p_max_pu = self.network.generators_t.p_max_pu[generator].iloc[time]
        except KeyError:
            p_max_pu = 1.0
        return self.network.generators.p_nom[generator] * p_max_pu

    def getGeneratorStatus(self, gen: str, solution: list, time: int = 0
                           ) -> bool:
        """
        Returns the status of a generator 'gen' (i.e. on or off) at a
        time slice 'time' in a given solution.

        Args:
            gen: (str)
                The label of the generator.
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            time: (int)
                Index of time slice for which to get the generator
                status.
        """
        return self.data[gen]['indices'][time] in solution

    def getGeneratorDictionary(self, solution: list,
                               stringify: bool = True) -> dict:
        """
        Builds a dictionary containing the status of all generators at
        all time slices for a given solution of qubit spins.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            stringify: (bool)
                If this is true, dictionary keys are cast to strings, so
                they can for json

        Returns:
            (dict)
                A dictionary containing the status of all generators at
                all time slices. The keys are either tuples of the
                label of the generator and the index of the time slice,
                or these tuples typecast to strings, depending on the
                'stringify' argument. The values are booleans, encoding
                the status of the generators at the time slice.
        """
        result = {}
        for generator in self.network.generators.index:
            for time in range(len(self.snapshots)):
                key = (generator, time)
                if stringify:
                    key = str(key)
                result[key] = int(
                    self.getGeneratorStatus(gen=generator, solution=solution,
                                            time=time))
        return result

    def getFlowDictionary(self, solution: list,
                          stringify: bool = True) -> dict:
        """
        Builds a dictionary containing all power flows at all time
        slices for a given solution of qubit spins.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            stringify: (bool)
                If this is true, dictionary keys are cast to strings, so
                they can for json

        Returns:
            (dict)
                A dictionary containing the flow of all lines at
                all time slices. The keys are either tuples of the
                label of the generator and the index of the time slice,
                or these tuples typecast to strings, depending on the
                'stringify' argument. The values are floats,
                representing the flow of the lines at the time slice.
        """
        result = {}
        for lineId in self.network.lines.index:
            for time in range(len(self.snapshots)):
                key = (lineId, time)
                if stringify:
                    key = str(key)
                result[key] = self.getEncodedValueOfComponent(
                    component=lineId, solution=solution, time=time)
        return result

    def getLoad(self, bus: str, time: int = 0, silent: bool = True) -> float:
        """
        Returns the total load at a bus at a given time slice.

        Args:
            bus: (str)
                Label of bus at which to calculate the total load.
            time: (int)
                Index of time slice for which to get the total load.

        Returns:
            (float)
                The total load at 'bus' at time slice 'time'.
        """
        loadsAtCurrentBus = self.network.loads[
            self.network.loads.bus == bus
            ].index
        allLoads = self.network.loads_t['p_set'].iloc[time]
        result = allLoads[allLoads.index.isin(loadsAtCurrentBus)].sum()
        if result == 0:
            if not silent:
                print(f"Warning: No load at {bus} at timestep {time}.\n"
                      f"Falling back to constant load")
            allLoads = self.network.loads['p_set']
            result = allLoads[allLoads.index.isin(loadsAtCurrentBus)].sum()
        if result < 0:
            raise ValueError("negative Load at current Bus")
        return result

    def getTotalLoad(self, time: int) -> float:
        """
        Returns the total load over all buses at one time slice.

        Args:
            time: (int)
                Index of time slice at which to return the load.
        Returns:
            (float)
                The total load of the network at time slice `time`.
        """
        load = 0.0
        for bus in self.network.buses.index:
            load += self.getLoad(bus, time)
        return load

    def getRepresentingQubits(self, component: str, time: int = 0) -> list:
        """
        Returns a list of all qubits that are used to encode a network
        component at a given time slice.
        A component is identified by a string assumed to be encoded in
        one block with constant encoding size per time slice and order
        of time slices being respected in the encoding.

        Args:
            component: (str)
                Label of the network component.
            time: (int)
                Index of time slice for which to get representing
                qubits.

        Returns:
            (list)
                List of integers which are qubits that represent the
                component.
        """
        encodingLength = self.data[component]["encodingLength"]
        return self.data[component]["indices"][
               time * encodingLength: (time + 1) * encodingLength]

    def getQubitMapping(self, time: int = 0) -> dict:
        """
        Returns a dictionary with all network components and which
        qubits were used for representation in an Ising spin glass
        problem.
        
        Args:
            time: (int)
                Index of time slice for which to get qubit map

        Returns:
            (dict)
                Dictionary of all network components and their qubits.
                Network components labels are the keys and the values
                are the ranges of qubits used for their encoding.
        """
        return {component: self.getRepresentingQubits(component=component,
                                                      time=time)
                for component in self.data.keys()
                if isinstance(component, str)}

    def getInteraction(self, *args) -> float:
        """
        Returns the interaction coefficient of a list of qubits.
        
        Args:
            args: (int)
                All qubits that are involved in this interaction.

        Returns:
            (float)
                The interaction strength between all qubits in args.
        """
        sortedUniqueArguments = tuple(sorted(set(args)))
        return self.problem.get(sortedUniqueArguments, 0.0)

    def getEncodedValueOfComponent(self, component: str, solution: list,
                                   time: int = 0) -> float:
        """
        Returns the encoded value of a component according to the spin
        configuration in solution at a given time slice.
        A component is represented by a list of weighted qubits. The
        encoded value is the weighted sum of all active qubits.

        Args:
            component: (str)
                Label of the network component for which to retrieve the
                encoded value.
            solution: (list)
                List of all qubits which have spin -1 in the solution.
            time: (int)
                Index of time slice for which to retrieve the encoded
                value.

        Returns:
            (float)
                Value of the component encoded in the spin configuration
                of solution.
        """
        value = 0.0
        encodingLength = self.data[component]["encodingLength"]
        for idx in range(time * encodingLength, (time + 1) * encodingLength,
                         1):
            if self.data[component]['indices'][idx] in solution:
                value += self.data[component]['weights'][idx]
        return value

    def generateReport(self, solution: list) -> dict:
        """
        For the given solution, calculates various properties of the
        solution.

        Args:
            solution: (list)
                List of all qubits that have spin -1 in a solution.
        Returns:
            (dict)
                A dicitionary containing general information about the
                solution.
        """
        return {
            "totalCost": self.calcCost(solution=solution),
            "kirchhoffCost": self.calcKirchhoffCost(solution=solution),
            "powerImbalance": self.calcPowerImbalance(solution=solution),
            "totalPower": self.calcTotalPowerGenerated(solution=solution),
            "marginalCost": self.calcMarginalCost(solution=solution),
            "individualKirchhoffCost": self.individualCostContribution(
                solution=solution),
            "unitCommitment": self.getGeneratorDictionary(solution=solution,
                                                          stringify=True),
            "powerflow": self.getFlowDictionary(solution=solution,
                                                stringify=True),
            "hamiltonian": self.getHamiltonianMatrix()
        }

    def calcCost(self, solution: list,
                 isingInteractions: dict = None) -> float:
        """
        Calculates the energy of a spin state including the constant
        energy contribution.
        The default Ising spin glass state that is used to calculate
        the energy of a solution is the full problem stored in the
        IsingBackbone. Ising subproblems can overwrite which Ising
        interactions are used to calculate the energy to get subproblem
        specific information. The assignemt of qubits to the network is
        still fixed.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            isingInteractions: (dict)
                The Ising problem to be used to calculate the energy.

        Returns:
            (float)
                The energy of the spin glass state in solution.
        """
        solution = set(solution)
        if isingInteractions is None:
            isingInteractions = self.problem
        totalCost = 0.0
        for spins, weight in isingInteractions.items():
            if len(spins) == 1:
                factor = 1
            else:
                factor = -1
            for spin in spins:
                if spin in solution:
                    factor *= -1
            totalCost += factor * weight
        return totalCost

    def individualMarginalCost(self, solution: list) -> dict:
        """
        Returns a dictionary which contains the marginal cost incurred
        at every bus at every time slice.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.

        Returns:
            (dict)
                A dictionary containing the marginal costs incurred at
                every bus at every time slice. The keys are tuples of
                the bus name and the index of the time slice and the
                values are the marginal costs at this bus at this time
                slice.
        """
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib,
                       **self.calcMarginalCostAtBus(bus=bus,
                                                    solution=solution)
                       }
        return contrib

    def calcMarginalCostAtBus(self, bus: str, solution: list) -> dict:
        """
        Returns a dictionary which contains the marginal cost incurred
        at 'bus' at every time slice.

        Args:
            bus: (str)
                Label of the bus.
            solution: (list)
                A list of all qubits which have spin -1 in the solution.

        Returns:
            (dict)
                A dictionary containing the marginal costs incurred at
                'bus' at every time slice. The keys are tuples of the
                'bus' and the index of the time slice, typecast to
                strings, and the values are the marginal costs at this
                bus at this time slice.
        """
        contrib = {}
        for time in range(len(self.snapshots)):
            marginalCost = 0.0
            components = self.getBusComponents(bus)
            for generator in components['generators']:
                if self.getGeneratorStatus(gen=generator,
                                           solution=solution,
                                           time=time):
                    marginalCost \
                        += self.network.generators["marginal_cost"].loc[
                               generator] \
                           * self.data[generator]['weights'][0]
            contrib[str((bus, time))] = marginalCost
        return contrib

    def calcMarginalCost(self, solution: list) -> float:
        """
        Calculate the total marginal cost incurred by a solution.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.

        Returns:
            (float)
                The total marginal cost incurred without monetaryFactor
                scaling.
        """
        marginalCost = 0.0
        for key, val in self.individualMarginalCost(solution=solution).items():
            marginalCost += val
        return marginalCost

    # getter for encoded ising problem parameters
    def siquanFormat(self) -> list:
        """
        Returns the complete problem in the format required by the
        siquan solver.

        Returns:
            (list)
                A list of tuples of the form (interaction-coefficient,
                list(qubits)).
        """
        return [(v, list(k)) for k, v in self.problem.items() if
                v != 0 and len(k) > 0]

    def getHamiltonianMatrix(self) -> list:
        """
        Returns a matrix containing the Ising hamiltonian
        
        Returns:
            (list)
                A list of lists representing the hamiltonian matrix.
        """
        qubits = range(self.allocatedQubits)
        hamiltonian = [
            [self.getInteraction(i, j) for i in qubits] for j in qubits
        ]
        return hamiltonian

    def getHamiltonianEigenvalues(self) -> np.ndarray:
        """
        Returns the eigenvalues and normalized eigenvectors of the
        hamiltonian matrix.
        
        Returns:
            (np.ndarray)
                A numpy array containing all eigenvalues.
        """
        return np.linalg.eigh(self.getHamiltonianMatrix())


class AbstractIsingSubproblem:
    """
    An interface for classes that model the Ising formulation
    subproblem of an unit commitment problem.
    Classes that model a subproblem/constraint are subclasses of this
    class and adhere to the following structure. Each subproblem/
    constraint corresponds to one class. This class has a factory method
    which chooses the correct subclass of itself. Any of those have a
    method that accepts an IsingBackbone as the argument and encodes
    it's problem onto that object.
    """

    def __init__(self, backbone: IsingBackbone, config: dict):
        """
        The constructor for a subproblem to be encode into the Ising
        subproblem.
        Different formulations of the same subproblems use a (factory)
        classmethod to choose the correct subclass and call this
        constructor. The attributes set here are the minimal attributes
        that are expected.
        
        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance.
        """
        self.problem = {}
        self.scaleFactor = config["scaleFactor"]
        self.backbone = backbone
        self.network = backbone.network

    @classmethod
    def buildSubproblem(cls, backbone: IsingBackbone,
                        configuration: dict) -> 'AbstractIsingSubproblem':
        """
        Returns an instance of the class set up according to the
        configuration.
        This is done by choosing the corresponding subclass of the
        configuration. After initialization, the instance can encode
        this subproblem into the isingBackbone by calling the
        encodeSubproblem method.

        Args:
            backbone: (IsingBackbone)
                The isingBackbone used to encode the subproblem.
            configuration: (dict)
                The configuration dictionary, containing all data
                necessary to initalize.

        Returns:
            (AbstractIsingSubproblem)
                The constructed Ising subproblem.
        """
        raise NotImplementedError

    def encodeSubproblem(self):
        """
        This encodes the problem an instance of a subclass is describing
        into the isingBackbone instance. After this call, the
        corresponding QUBO is stored in the isingBackbone.
        """
        raise NotImplementedError

    def calcCost(self, solution: list) -> float:
        """
        Calculates the energy of a spin state including the constant
        energy contribution by delegating it to the IsingBackbone.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.

        Returns:
            (float)
                The energy of the spin glass state.
        """
        return self.backbone.calcCost(solution=solution,
                                      isingInteractions=self.problem)


class MarginalCostSubproblem(AbstractIsingSubproblem):
    @classmethod
    def buildSubproblem(cls, backbone: IsingBackbone,
                        configuration: dict) -> 'MarginalCostSubproblem':
        """
        A factory method for obtaining the marginal cost model specified
        in configuration.
        Returns an instance of the class set up according to the
        configuration.
        This is done by choosing the corresponding subclass of the
        configuration. After initialization, the instance can encode
        this subproblem into the isingBackbone by calling the
        encodeSubproblem method.

        Args:
            backbone: (IsingBackbone)
                The isingBackbone used to encode the subproblem.
            configuration: (dict)
                The configuration dictionary, containing all data
                necessary to initalize.

        Returns:
            (MarginalCostSubproblem)
                The constructed Ising instance ready to encode into the
                backbone.
        """
        subclassTable = {
            "GlobalCostSquare": GlobalCostSquare,
            "GlobalCostSquareWithSlack": GlobalCostSquareWithSlack,
            "MarginalAsPenalty": MarginalAsPenalty,
            "LocalMarginalEstimation": LocalMarginalEstimation,
        }
        return subclassTable[configuration.pop("formulation")](
            backbone=backbone, config=configuration)


class MarginalAsPenalty(MarginalCostSubproblem):
    """
    A subproblem class that models the minimization of the marginal
    costs. It does this by adding a penalty to each qubit of a generator
    with the value being the marginal costs incurred by committing that
    generator. This linear penalty can be slightly changed.
    """

    def __init__(self, backbone: IsingBackbone, config: dict):
        """
        The constructor for encoding marginal cost as linear penalties.
        It inherits its functionality from the AbstractIsingSubproblem
        constructor. Additionally it sets three more parameters which
        slightly change how the penalty is applied:
            `offsetEstimationFactor`: sets an offset across all generators
                by the cost of the most efficient generator scaled by this
                factor
            `estimatedCostFactor`: is going to be removed
            `offsetBuildFactor`: is going to be removed

        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance.
        """
        super().__init__(backbone, config)
        # factor which in conjuction with the minimal cost per energy produced
        # describes by how much each marginal cost per unit procuded is offset
        # in all generators to bring the average cost of a generator closer to
        # zero
        self.offsetEstimationFactor = float(config["offsetEstimationFactor"])
        # factor to scale estimated cost at a bus after calculation
        self.estimatedCostFactor = float(config["estimatedCostFactor"])
        # factor to scale marginal cost of a generator when constructing ising
        # interactions
        self.offsetBuildFactor = float(config["offsetBuildFactor"])

    def encodeSubproblem(self) -> None:
        """
        Encodes the minimization of the marginal cost by applying a
        penalty at each qubit of a generator equal to the cost it would
        incur if committed.

        Returns:
            (None)
                Modifies self.backbone.
        """
        # Marginal costs are only modelled as linear penalty. Thus, it
        # suffices to iterate over all time steps and all buses to get all
        # generators
        for time in range(len(self.network.snapshots)):
            for bus in self.network.buses.index:
                self.encodeMarginalCosts(bus=bus, time=time)

    def marginalCostOffset(self) -> float:
        """
        Returns a float by which all generator marginal costs per power
        will be offset. Since every generator will be offset, this will
        not change relative costs between them. It changes the range of
        energy contributions this constraint provides. Adding marginal
        costs as a cost to the QUBO formulation will penalize all
        generator configurations. The offset shifts it, so that the
        cheapest generator doesn't get any penalty.

        Returns:
            (float)

        @return: float
            a float that in is in the range of generator marginal costs
        """
        return 1.0 * min(self.network.generators[
                             "marginal_cost"]) * self.offsetEstimationFactor

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
        generators = self.backbone.getBusComponents(bus)['generators']
        costOffset = self.marginalCostOffset()
        for generator in generators:
            self.backbone.coupleComponentWithConstant(
                generator,
                couplingStrength=self.scaleFactor * \
                                 (self.network.generators["marginal_cost"].loc[
                                      generator] - \
                                  costOffset),
                time=time
            )


class LocalMarginalEstimation(MarginalCostSubproblem):
    """
    A subproblem class that models the minimization of the marginal costs. It does this
    by estimation the cost at each bus and models a minimization of the distance of the
    incurred cost to the estimated cost.
    """

    def __init__(self, backbone, config):
        """
        A constructor for encoding marginal cost as linear penalties
    
        This sets three additional parameters which slightly change how the penalty is applied
        `offsetEstimationFactor`: sets an offset across all generators by the cost of the most
                                efficient generator scaled by this factor
        `estimatedCostFactor`: is going to be removed
        `offsetBuildFactor`: is going to be removed
        Args:
            backbone: (IsingBackbone) The backbone to use for encoding network components with qubits
            configuration: (dict) A dictionary containing all data necessary to initalize
        """
        super().__init__(backbone, config)
        self.offsetEstimationFactor = float(config["offsetEstimationFactor"])
        # factor to scale estimated cost at a bus after calculation
        self.estimatedCostFactor = float(config["estimatedCostFactor"])
        # factor to scale marginal cost of a generator when constructing ising
        # interactions
        self.offsetBuildFactor = float(config["offsetBuildFactor"])

    def encodeSubproblem(self, isingBackbone: IsingBackbone, ):
        """
        Encodes the minimization of the marginal cost.

        It estimates the cost at each bus and models a minimization of the distance of the
        incurred cost to the estimated cost. The exact modeling of this can be adjusted
        using the parameters: ``

        Args:
            isingBackbone: (IsingBackbone) isingBackbone on which qubits the marginal cost will be encoded
        Returns:
            (None) modifies `isingBackone`
        """
        # Estimation is done independently at each bus. Thus it suffices to iterate over all snapshots
        # and buses to encode the subproblem
        for time in range(len(self.network.snapshots)):
            for bus in self.network.buses.index:
                self.encodeMarginalCosts(bus, time)

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
        marginalCostList = [self.network.generators["marginal_cost"].loc[gen]
                            for gen in sortedGenerators]
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
        remainingLoad = self.backbone.getLoad(bus, time)
        generators = self.backbone.getBusComponents(bus)['generators']
        sortedGenerators = sorted(
            generators,
            key=lambda gen: self.network.generators["marginal_cost"].loc[gen]
        )
        offset = self.chooseOffset(sortedGenerators)
        costEstimation = 0.0
        for generator in sortedGenerators:
            if remainingLoad <= 0:
                break
            suppliedPower = min(remainingLoad,
                                self.backbone.data[generator]['weights'][0])
            costEstimation += suppliedPower * (
                        self.network.generators["marginal_cost"].loc[
                            generator] - offset)
            remainingLoad -= suppliedPower
        return costEstimation, offset

    def calculateCost(self, componentToBeValued, allComponents, offset,
                      estimatedCost, load, bus):
        if componentToBeValued in allComponents['generators']:
            return self.network.generators["marginal_cost"].loc[
                       componentToBeValued] - offset
        if componentToBeValued in allComponents['positiveLines']:
            return 0.5 * estimatedCost / load
        if componentToBeValued in allComponents['negativeLines']:
            return 0.5 * - estimatedCost / load

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
        components = self.backbone.getBusComponents(bus)
        flattenedComponenents = components['generators'] + \
                                components['positiveLines'] + \
                                components['negativeLines']

        estimatedCost, offset = self.estimateMarginalCostAtBus(bus, time)
        estimatedCost *= self.estimatedCostFactor
        offset *= self.offsetBuildFactor
        load = self.backbone.getLoad(bus, time)

        self.backbone.addInteraction(0.25 * estimatedCost ** 2)
        for firstComponent in flattenedComponenents:
            self.backbone.coupleComponentWithConstant(
                firstComponent,
                - 2.0 * self.calculateCost(firstComponent, components, offset,
                                           estimatedCost, load, bus) * \
                estimatedCost * \
                self.scaleFactor
            )
            for secondComponent in flattenedComponenents:
                curFactor = self.scaleFactor * \
                            self.calculateCost(firstComponent, components,
                                               offset, estimatedCost, load,
                                               bus) * \
                            self.calculateCost(secondComponent, components,
                                               offset, estimatedCost, load,
                                               bus)
                self.backbone.coupleComponents(
                    firstComponent,
                    secondComponent,
                    couplingStrength=curFactor
                )


class GlobalCostSquare(MarginalCostSubproblem):
    def __init__(self, backbone, config):
        """
        A constructor for encoding marginal cost as linear penalties
    
        This sets three additional parameters which slightly change how the penalty is applied
        `offsetEstimationFactor`: sets an offset across all generators by the cost of the most
                                efficient generator scaled by this factor
        `estimatedCostFactor`: is going to be removed
        `offsetBuildFactor`: is going to be removed
        Args:
            backbone: (IsingBackbone) The backbone to use for encoding network components with qubits
            configuration: (dict) A dictionary containing all data necessary to initalize
        """
        super().__init__(backbone, config)
        self.offsetEstimationFactor = float(config["offsetEstimationFactor"])
        # factor to scale estimated cost at a bus after calculation
        self.estimatedCostFactor = float(config["estimatedCostFactor"])
        # factor to scale marginal cost of a generator when constructing ising
        # interactions
        self.offsetBuildFactor = float(config["offsetBuildFactor"])

    def printEstimationReport(self, estimatedCost, offset, time):
        """
        prints the estimated marginal cost and the offset of the cost per MW produced
    
        Args:
            estimatedCost: (float) TODO
            offset: (float) TODO
            time: (int) index of the current time step
        Returns:
            (None) prints to stdout
        """
        print(f"--- Estimation Parameters at timestep {time} ---")
        print(f"Absolute offset: {offset}")
        print(f"Minimal estimated Cost (with offset): {estimatedCost}")
        print(
            f"Current total estimation at {time}: {offset * self.backbone.getTotalLoad(time)}")
        print("---")

    def encodeSubproblem(self, isingBackbone: IsingBackbone, ):
        for time in range(len(self.network.snapshots)):
            self.encodeMarginalCosts(time)

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
        marginalCostList = [self.network.generators["marginal_cost"].loc[gen]
                            for gen in sortedGenerators]
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
            load += self.backbone.getLoad(bus, time)

        sortedGenerators = sorted(
            self.network.generators.index,
            key=lambda gen: self.network.generators["marginal_cost"].loc[gen]
        )
        offset = self.chooseOffset(sortedGenerators)
        costEstimation = 0.0
        for generator in sortedGenerators:
            if load <= 0:
                break
            suppliedPower = min(load,
                                self.backbone.data[generator]['weights'][0])
            costEstimation += suppliedPower * (
                        self.network.generators["marginal_cost"].loc[
                            generator] - offset)
            load -= suppliedPower
        return costEstimation + expectedAdditonalCost, offset

    # TODO refactor using a isingbackbone function for encoding squared distances
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
        # TODO make this more readable
        estimatedCost, offset = self.estimateGlobalMarginalCost(time,
                                                                expectedAdditonalCost=0)
        self.printEstimationReport(estimatedCost, offset, time)
        generators = self.network.generators.index
        # estimation of marginal costs is a global estimation. Calculate total power needed
        load = self.backbone.getTotalLoad(time)
        # offset the marginal costs per energy produces and encode problem into backbone
        for gen1 in generators:
            marginalCostGen1 = self.network.generators["marginal_cost"].loc[
                                   gen1] - offset
            for gen2 in generators:
                marginalCostGen2 = \
                self.network.generators["marginal_cost"].loc[gen2] - offset
                curFactor = self.scaleFactor * \
                            marginalCostGen1 * \
                            marginalCostGen2
                self.backbone.coupleComponents(
                    gen1,
                    gen2,
                    couplingStrength=curFactor
                )


class KirchhoffSubproblem(AbstractIsingSubproblem):

    def __init__(self, backbone, config):
        super().__init__(backbone, config)

    @classmethod
    def buildSubproblem(cls, backbone, configuration) -> [str,
                                                          'AbstractIsingSubproblem']:
        """
        returns the name of the subproblem and an instance of the class set up according to the configuration.
        This is done by choosing the corresponding subclass of the configuration.
        After initialization, the instance can encode this subproblem into an isingBackbone by calling
        the encodeSubproblem method
        """
        return KirchhoffSubproblem(backbone, configuration)

    def encodeSubproblem(self, isingBackbone: IsingBackbone, ):
        """
        Encodes the kirchhoff constraint at each bus
    
        Args:
            isingBackbone: (IsingBackbone) isingBackbone on which qubits the kirchhoff 
                                        constraint will be encoded
            configuration: (dict) contains the scale factor of the kirchhoff constraint
        Returns:
            (None) Constructs the ising problem for the kirchhoff cost and encodes.
        """
        for time in range(len(isingBackbone.snapshots)):
            for node in isingBackbone.network.buses.index:
                self.encodeKirchhoffConstraint(isingBackbone, node, time)
        self.problem = isingBackbone.cachedProblem

    def encodeKirchhoffConstraint(self, isingBackbone, bus, time=0):
        """
        Adds the kirchhoff constraint at a bus to the problem formulation. 

        The kirchhoff constraint is that the sum of all power generating elements 
        (generators, power flow towards the bus) is equal to the sum of all 
        load generating elements (bus specific load, power flow away from the bus).
        Deviation from equality is penalized quadratically 

        At a bus, the total power can be calculated as:
        (-Load + activeGenerators + powerflowTowardsBus - powerflowAwayFromBus) ** 2
        The function expands this expression and adds all result products of two components
        by looping over them
        @param bus: str
            label of the bus at which to enforce the kirchhoff constraint
        @param time: int
            index of time slice at which to enforce the kirchhoff contraint
        @return: None
            modifies self.problem. Adds to previously written interaction cofficient
        """
        components = isingBackbone.getBusComponents(bus)
        flattenedComponenents = components['generators'] + \
                                components['positiveLines'] + \
                                components['negativeLines']
        demand = isingBackbone.getLoad(bus, time=time)

        # constant load contribution to cost function so that a configuration that fulfills the
        # kirchhoff contraint has energy 0
        isingBackbone.addInteraction(self.scaleFactor * demand ** 2)
        for component1 in flattenedComponenents:
            # this factor sets the the scale, aswell as the sign to encode if a active component
            # acts a generator or a load
            factor = self.scaleFactor
            if component1 in components['negativeLines']:
                factor *= -1.0
            # reward/penalty term for matching/adding load. Contains all products with the Load
            isingBackbone.coupleComponentWithConstant(component1,
                                                      - 2.0 * factor * demand)
            for component2 in flattenedComponenents:
                # adjust sing for direction of flow at line
                if component2 in components['negativeLines']:
                    curFactor = -factor
                else:
                    curFactor = factor
                # attraction/repulsion term for different/same sign of power at components
                isingBackbone.coupleComponents(component1, component2,
                                               couplingStrength=curFactor)

    def calcPowerImbalanceAtBus(self, bus, result, silent=True):
        """
        returns a dictionary containg the absolute values of the power 
        imbalance/mismatch at a bus for each time step
        
        Args:
            bus: (str) label of the bus at which to calculate power imbalance
            result: (list) list of all qubits which have spin -1 in the solution 

        Returns:
            (dict) dictionary with keys of the type (str, int) over all  time
                        slices and the string alwyays being the chosen bus
        """
        contrib = {}
        # TODO option to take only some snapshots
        for t in range(len(self.network.snapshots)):
            load = - self.backbone.getLoad(bus, t)
            components = self.backbone.getBusComponents(bus)
            for gen in components['generators']:
                load += self.backbone.getEncodedValueOfComponent(gen, result,
                                                                 time=t)
            for lineId in components['positiveLines']:
                load += self.backbone.getEncodedValueOfComponent(lineId,
                                                                 result,
                                                                 time=t)
            for lineId in components['negativeLines']:
                load -= self.backbone.getEncodedValueOfComponent(lineId,
                                                                 result,
                                                                 time=t)
            if load and not silent:
                print(f"Imbalance at {bus}::{load}")
            contrib[str((bus, t))] = load
        return contrib

    def calcTotalPowerGeneratedAtBus(self, bus, solution, time=0):
        """
        Calculates how much power is generated using generators at this bus at a time step
    
        Ignores any power flow or load.
        Args:
            bus: (str) label of the bus at which to calculate total power 
            solution: (list) list of all qubits that have spin -1 in a solution
            time: (int) index of the time step at which to calculate total power
        Returns:
            (float) the total power generated without flow or loads
        """
        totalPower = 0.0
        generators = self.backbone.getBusComponents(bus)['generators']
        for generator in generators:
            totalPower += self.backbone.getEncodedValueOfComponent(generator,
                                                                   solution,
                                                                   time=time)
        return totalPower

    def calcTotalPowerGenerated(self, solution, time=0):
        """
        Calculates how much power is generated using generators across the entire network at a time step
    
        Args:
            solution: (list) list of all qubits that have spin -1 in a solution
            time: (int) index of the time step at which to calculate total power
        Returns:
            (float) the total power generated without flow or loads
        """
        totalPower = 0.0
        for bus in self.network.buses.index:
            totalPower += self.calcTotalPowerGeneratedAtBus(bus, solution,
                                                            time=time)
        return totalPower

    def calcPowerImbalance(self, solution):
        """
        returns the sum of all absolutes values of power imbalances at each bus over all time steps
        This is basically like the kirchhoff cost except with a linear penalty
        
        Args:
            solution: (list) list of all qubits which have spin -1 in the solution
        Returns:
            (float) the sum of all absolute values of every ower imbalance at every bus
        """
        powerImbalance = 0.0
        for bus in self.network.buses.index:
            for _, imbalance in self.calcPowerImbalanceAtBus(bus,
                                                             solution).items():
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
            key: (imbalance * self.scaleFactor) ** 2
            for key, imbalance in
            self.calcPowerImbalanceAtBus(bus, result, silent=silent).items()
        }

    def calcKirchhoffCost(self, solution: list):
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

    def individualCostContribution(self, solution, silent=True):
        """
        returns a dictionary which contains the kirchhoff cost incurred at every bus at
        every time slice scaled by the KirchhoffFactor

        @param solution: list
           list of all qubits which have spin -1 in the solution 
        @return: dict
            dictionary with keys of the form (str,int) over all busses and time slices
        """
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib, **self.calcKirchhoffCostAtBus(bus, solution,
                                                                silent=silent)}
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
            key: imbalance ** 2
            for key, imbalance in
            self.individualPowerImbalance(bus, result, silent=silent).items()
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
            contrib = {**contrib, **self.calcPowerImbalanceAtBus(bus, solution,
                                                                 silent=silent)}
        return contrib


class GlobalCostSquareWithSlack(GlobalCostSquare):
    """
    A subproblem class that models the minimization of the marginal costs. It does this
    by estimating it and then modelling the squared distance of the actual cost to the
    estimated cost. It also adds a slack term to the estimation which is independent
    of the network and serves to slightly adjust the estimation during the optimization
    """
    # a dict to map config strings to functions which are used creating lists of numbers, which
    # can be used for weights of slack variables
    slackRepresentationDict = {
        "binaryPower": binaryPower,
    }

    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        slackWeightGenerator = self.slackRepresentationDict[
            config.get("slackType", "binaryPower")]
        # an additional factor for scaling the weights of the qubits acting as slack variables
        slackScale = config.get("slackScale", 1.0)
        # number of slack qubits used
        slackSize = config.get("slackSize", 7)
        slackWeights = [- slackScale * i for i in
                        slackWeightGenerator(slackSize)]
        # adding slack qubits with the label `slackMarginalCost`
        self.backbone.createQubitEntriesForComponent(
            "slackMarginalCost",
            weights=slackWeights * len(backbone.snapshots),
            encodingLength=len(slackWeights)
        )

    # TODO refactor using a isingbackbone function for encoding squared distances
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
        estimatedCost, offset = self.estimateGlobalMarginalCost(time,
                                                                expectedAdditonalCost=0)
        self.printEstimationReport(estimatedCost, offset, time)
        generators = self.network.generators.index
        generators = list(generators) + ["slackMarginalCost"]
        load = 0.0
        for bus in self.network.buses.index:
            load += self.backbone.getLoad(bus, time)
        for gen1 in generators:
            if gen1 == "slackMarginalCost":
                marginalCostGen1 = 1.
            else:
                marginalCostGen1 = \
                self.network.generators["marginal_cost"].loc[gen1] - offset
            for gen2 in generators:
                if gen2 == "slackMarginalCost":
                    marginalCostGen2 = 1.
                else:
                    marginalCostGen2 = \
                    self.network.generators["marginal_cost"].loc[gen2] - offset
                curFactor = self.scaleFactor * \
                            marginalCostGen1 * \
                            marginalCostGen2
                self.backbone.coupleComponents(
                    gen1,
                    gen2,
                    couplingStrength=curFactor
                )


class StartupShutdown(AbstractIsingSubproblem):
    pass
#    def encodeStartupShutdownCost(self, bus, time=0):
#        """
#        Adds the startup and shutdown costs for every generator attached to the bus. Those
#        costs are monetary costs incurred whenever a generator changes its status from one
#        time slice to the next. The first time slice doesn't incurr costs because the status
#        of the generators before is unknown
#        
#        @param bus: str
#            label of the bus at which to add startup and shutdown cost
#        @param time: int
#            index of time slice which contains the generator status after a status change
#        @return: None
#            modifies self.problem. Adds to previously written interaction cofficient 
#        """
#        # no previous information on first time step or when out of bounds

#        if time == 0 or time >= len(self.snapshots):
#            return
#
#        generators = self.getBusComponents(bus)['generators']
#
#        for generator in generators:
#            startup_cost = self.network.generators["start_up_cost"].loc[generator]
#            shutdown_cost = self.network.generators["shut_down_cost"].loc[generator]
#
#            # start up costs
#            # summands of (1-g_{time-1})  * g_{time})
#            self.coupleComponentWithConstant(
#                    generator,
#                    couplingStrength=self.monetaryCostFactor * startup_cost,
#                    time=time
#            )
#            self.coupleComponents(
#                    generator,
#                    generator,
#                    couplingStrength= -self.monetaryCostFactor * startup_cost,
#                    time = time,
#                    additionalTime = time -1
#            )
#
#            # shutdown costs
#            # summands of g_{time-1} * (1-g_{time})
#            self.coupleComponentWithConstant(
#                    generator,
#                    couplingStrength=self.monetaryCostFactor * shutdown_cost,
#                    time=time-1
#            )
#            self.coupleComponents(
#                    generator,
#                    generator,
#                    couplingStrength= -self.monetaryCostFactor * shutdown_cost,
#                    time = time,
#                    additionalTime = time -1
#            )
