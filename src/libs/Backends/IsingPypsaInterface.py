"""This module is the central piece for the mathematical
model of the unit commitment problem as a QUBO/ising problem
It provides two core components: 
- The IsingBackbone class which serves as a layer of abstraction
    between the pypsa network and the groups of qubits used to
    represent the components of the network. It also provides
    methods to couple groups of qubits together and to translate
    the state of qubits into the corresponding network
- The AbstractIsingSubproblem class, which defines an interface for
    classes, that model a constraint or the optimization goal
    of the unit commitment problem. For that, an instance of an
    AbstractIsingSubproblem models it corresponding problem/constraint
    using the methods and abstractions provided by an IsingBackbone.

Expanding on the type of networks and problems that can be read can
be archieved by extending the IsingBackbone. New constraint can be
added by writing a new class that conforms to the AbstractIsingSubproblem.
"""

from abc import ABC

import numpy as np

import pypsa


class ComponentToQubitEncoder:
    """An interface for classes that, when encoding a network component into
    multiple qubits, calculate the weights used to do so
    """
    def __init__(self, network):
        """
        This sets up which pypsa network to use when encoding lines
        and generators
    
        Args:
            network: (pypsa.Network) the pypsa network which's component to 
                encode into qubits
        """
        self.network = network

    @classmethod
    def build_qubit_encoder(cls, generator_representation, line_representation):
        """
        A factory used for generating the 
    
        Args:
            PAR
        Returns:
            (type) description
        """
        pass
    


def integer_decomposition_powers_of_two_and_rest(number: int):
    """
    for an integer, constructs a list of powers of two + a rest, such 
    that the sum over that list is equal to this number. This only
    uses positive numbers

    Args:
        number: (int) 
            the number to be decomposed into (mostly) powers of two
    Returns:
        (list)
            a list of integers with sum equal to number
    """
    if number == 0:
        return []
    bit_length = number.bit_length()
    positive_powers = [1 << idx for idx in range(bit_length - 1)]
    already_filled = (1 << bit_length - 1) - 1
    return positive_powers + [number - already_filled]


def single_qubit(number: int):
        """
        wraps the input number into a list with it as the single entry. 
        This list is used as a the qubits weight for a network generator
        which means it can either produce full or no power
    
        Args:
            number: (int) 
                a number which represents the output of the generator which
                to calculate qubit weights
        Returns:
            (list) a list with only the number as the only entry
        """
        return [number]


def cut_powers_of_two(capacity: float) -> list:
    """
    A method for splitting up the capacity of a line with a given
    maximum capacity.
    It uses powers of two to decompose the capacity and cuts off the 
    the biggest power of two so the total sum of all powers equals
    the capacity
    
    Args:
        capacity: (int)
            The capacity of the line to be decomposed.
    Returns:
        (list)
            A list of weights to be used in decomposing a line.
    """
    integer_capacity = int(capacity)
    positive_capacity = integer_decomposition_powers_of_two_and_rest(integer_capacity)
    negative_capacity = [- number for number in positive_capacity]
    return positive_capacity + negative_capacity


def fullsplit(capacity: int) -> list:
    """
    A method for splitting up the capacity of a line with a given
    maximum capacity.
    A line is split into qubits with weights just that any sum of
    active qubits never exceeds the capacity and so that it can
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


def binary_power(number: int) -> list:
    """
    return a cut-off binary representation of the argument. It is a list
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
    problem/constraint. 

    Modeling of various constraints is delegated
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
    generator_to_qubits_method_lookup_dict = {
        "single_qubit": single_qubit,
        "integer_decomposition": integer_decomposition_powers_of_two_and_rest
    }
    # The functions on line_to_qubits_method_lookup_dict define how some value of a capacity of
    # a transmission line is translated into qubits
    line_to_qubits_method_lookup_dict = {
        "cutpowersoftwo": cut_powers_of_two,
        "fullsplit": fullsplit,
        "binarysplit": binarysplit,
        "customsplit": customsplit,
    }

    def __init__(self, 
            network: pypsa.Network, 
            generator_to_qubits_method_name: str,
            line_to_qubits_method_name: str,
            configuration: dict):
        """
        Constructor for an Ising Backbone. It requires a network and
        the name of the function that defines how to encode lines. Then
        it goes through the configuration dictionary and encodes all
        sub-problem present into the instance.

        Args:
            network: (pypsa.Network)
                The pypsa network which to encode into qubits.
            line_to_qubits_method_name: (str)
                The name of the linesplit function as given in
                line_to_qubits_method_name_lookup_dict.
            configuration: (dict)
                A dictionary containing all subproblems to be encoded
                into an ising problem.
        """
        print()
        print("--- Generating Ising problem ---")
        self.subproblem_table = {
            "kirchhoff": KirchhoffSubproblem,
            "marginal_cost": MarginalCostSubproblem
        }
        if "kirchhoff" not in configuration:
            print("No Kirchhoff configuration found, "
                  "adding Kirchhoff constraint with Factor 1.0")
            configuration["kirchhoff"] = {"scale_factor": 1.0}

        # resolve string for splitting line capacty to function
        self.generator_to_qubits = IsingBackbone.generator_to_qubits_method_lookup_dict[generator_to_qubits_method_name]
        self.line_to_qubits = IsingBackbone.line_to_qubits_method_lookup_dict[line_to_qubits_method_name]

        # network to be solved
        self.network = network
        if "snapshots" in configuration:
            self.network.snapshots = self.network.snapshots[:configuration.pop("snapshots")]
        self.snapshots = network.snapshots

        # contains ising coefficients
        self.ising_coefficients = {}
        # mirrors encodings of `self.ising_coefficients`, but is reset after encoding a
        # subproblem to get ising formulations of subproblems
        self.cached_problem = {}

        # initializing data structures that encode the network into qubits
        # the encoding dict contains the mapping of network components to qubits
        self._qubit_encoding = {}
        # the weights dict containts a mapping of qubits to their weights
        self._qubit_weights = {}
        self.allocated_qubits = 0
        self.store_generators()
        self.store_lines()

        # read configuration dict, store in _subproblems and apply encodings
        self._subproblems = {}
        # dictionary of all support subproblems
        for subproblem, subproblem_configuration in configuration.items():
            if subproblem not in self.subproblem_table:
                print(f"{subproblem} is not a valid subproblem, skipping "
                      f"encoding")
                continue
            if not subproblem_configuration:
                print(f"Subproblem {subproblem} has no configuration data, "
                      f"skipping encoding")
                continue
            subproblem_instance = self.subproblem_table[
                subproblem].build_subproblem(self, subproblem_configuration)
            self._subproblems[subproblem] = subproblem_instance
            self.flush_cached_problem()
            subproblem_instance.encode_subproblem()
        print()
        print("--- Finish generating Ising Problem with the following subproblems ---")
        for key in self._subproblems:
            print("--- - " + key)

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
        unique_resolution = True
        for subproblem, subproblem_instance in self._subproblems.items():
            if hasattr(subproblem_instance, method_name):
                if unique_resolution:
                    unique_resolution = False
                    method = getattr(subproblem_instance, method_name)
                else:
                    raise AttributeError(f"{method_name} didn't resolve to "
                                         f"unique subproblem")
        if method:
            return method
        else:
            raise AttributeError(f"{method_name} was not found in any stored "
                                 f"subproblem")

    # obtain config file using a reader
    @classmethod
    def build_ising_problem(cls, network: pypsa.Network, config: dict):
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
        generator_to_qubits_method_name = config.pop("generator_representation", "single_qubit")
        line_to_qubits_method_name = config.pop("line_representation", "cutpowersoftwo")
        return IsingBackbone(network, 
                generator_to_qubits_method_name, 
                line_to_qubits_method_name,
                config)

    def flush_cached_problem(self) -> None:
        """
        Resets the cached changes of interactions.

        Returns:
            (None)
        """
        self.cached_problem = {}

    # functions to couple components. The couplings are interpreted as
    # multiplications of QUBO polynomials. The final interactions are
    # coefficients for an ising spin glass problem
    def add_interaction(self, *args) -> None:
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
                Modifies self.ising_coefficients by adding the strength of the
                interaction if an interaction coefficient is already
                set.
        """
        if len(args) > 3:
            raise ValueError(
                "Too many arguments for an interaction"
            )
        *key, interaction_strength = args
        key = tuple(sorted(key))
        for qubit in key:
            interaction_strength *= self._qubit_weights[qubit]

        # if we couple two spins, we check if they are different. If both spins
        # are the same, we substitute the product of spins with 1, since
        # 1 * 1 = -1 * -1 = 1 holds. This makes it into a constant
        # contribution. Doesn't work for higher order interactions
        if len(key) == 2:
            if key[0] == key[1]:
                key = tuple([])
        self.ising_coefficients[key] = self.ising_coefficients.get(key, 0) - interaction_strength
        self.cached_problem[key] = self.cached_problem.get(key, 0) - interaction_strength

    # TODO unify couple_components and with Constant
    def couple_component_with_constant(self, component: str,
                                       coupling_strength: float = 1,
                                       time: any = None) -> None:
        """
        Performs a QUBO multiplication involving a single variable on
        all qubits which are logically grouped to represent a component
        at a given time slice. This QUBO multiplication is translated
        into Ising interactions and then added to the currently stored
        Ising spin glass problem.

        Args:
            component: (str)
                Label of the network component.
            coupling_strength: (float)
                Coefficient of QUBO multiplication by which to scale the
                interaction. Does not contain qubit specific weight.
            time: (int)
                Index of time slice for which to couple qubit
                representing the component.

        Returns:
            (None)
                Modifies self.ising_coefficients. Adds to previously written
                interaction coefficient.
        """
        if time is None:
            print("falling back default time in constant")
            time = self.network.snapshots[0]
        component_adress = self.get_representing_qubits(component, time)
        for qubit in component_adress:
            # term with single spin after applying QUBO to Ising transformation
            self.add_interaction(qubit, 0.5 * coupling_strength)
            # term with constant cost contribution after applying QUBO to
            # Ising transformation
            self.add_interaction(0.5 * coupling_strength * self._qubit_weights[qubit])

    # TODO add a method to conveniently encode the squared distance to a fixed
    #  value into an ising

    def couple_components(self,
                          first_component: str,
                          second_component: str,
                          coupling_strength: float = 1,
                          time: any = None,
                          additional_time: int = None
                          ) -> None:
        """
        This method couples two labeled groups of qubits as a product
        according to their weight and the selected time step.
        It performs a QUBO multiplication involving exactly two
        components on all qubits which are logically grouped to
        represent these components at a given time slice. This QUBO
        multiplication is translated into Ising interactions, scaled by
        the coupling_strength and the respective weights of the qubits
        and then added to the currently stored Ising spin glass problem.

        Args:
            first_component: (str)
                Label of the first network component.
            second_component: (str)
                Label of the second network component.
            coupling_strength: (float)
                Coefficient of QUBO multiplication by which to scale all
                interactions.
            time: (int)
                Index of time slice of the first component for which to
                couple qubits representing it.
            additional_time: (int)
                Index of time slice of the second component for which
                to couple qubits representing it. The default parameter
                'None' is used if the time slices of both components
                are the same.
        Returns:
            (None)
                Modifies `self.ising_coefficients`. Adds to previously written
                interaction coefficient.

        Example:
            Let X_1, X_2 be the qubits representing firstComponent and
            Y_1, Y_2 the qubits representing second_component. The QUBO
            product the method translates into Ising spin glass
            coefficients is:
            (X_1 + X_2) * (Y_1 + Y_2) = X_1 * Y_1 + X_1 * Y_2
                                        + X_2 * Y_1 + X_2 * Y_2
        """
        # Replace None default values with their intended network component and
        # then figure out which qubits we want to couple based on the
        # component name and chosen time step
        if time is None:
            time = self.network.snapshots[0]
        if additional_time is None:
            additional_time = time
        first_component_adress = self.get_representing_qubits(first_component, time)
        second_component_adress = self.get_representing_qubits(second_component,
                                                               additional_time)
        # components with 0 weight (power, capacity) vanish in the QUBO
        # formulation
        if (not first_component_adress) or (not second_component_adress):
            return
        # retrieving corresponding qubits is done. Now perform qubo
        # multiplication by expanding the product and add each summand
        # individually.
        for first_qubit in first_component_adress:
            for second_qubit in second_component_adress:
                # The body of this loop corresponds to the multiplication of
                # two QUBO variables. According to the QUBO - Ising
                # translation rule x = (sigma+1)/2 one QUBO multiplication
                # results in 4 ising interactions, including constants

                # term with two spins after applying QUBO to Ising
                # transformation if both spin ids are the same, this will
                # add a constant cost.
                # add_interaction performs substitution of spin with a constant
                self.add_interaction(
                    first_qubit,
                    second_qubit,
                    coupling_strength * 0.25
                )
                # terms with single spins after applying QUBO to Ising
                # transformation
                self.add_interaction(
                    first_qubit,
                    coupling_strength * self._qubit_weights[second_qubit] * 0.25
                )
                self.add_interaction(
                    second_qubit,
                    coupling_strength * self._qubit_weights[first_qubit] * 0.25
                )
                # term with constant cost contribution after applying QUBO to
                # Ising transformation
                self.add_interaction(
                    self._qubit_weights[first_qubit]
                    * self._qubit_weights[second_qubit]
                    * coupling_strength * 0.25
                )

    # end of coupling functions

    def num_variables(self) -> int:
        """
        Returns how many qubits have already been used to model the
        problem components.
        When allocating qubits for a new component, those qubits will
        start at the value returned by this method and later updated.

        Returns:
            (int)
                The number of qubits already allocated.
        """
        return self.allocated_qubits

    def num_interactions(self) -> int:
        """
        Returns how many different non-zero interactions the ising
        problem has

        Returns:
            (int)
                number of ising interactions
        """
        return len(self.ising_coefficients)

    # create qubits for generators and lines
    def store_generators(self) -> None:
        """
        Assigns qubits to each generator in self.network. For each
        generator it writes generator specific parameters (i.e. power,
        corresponding qubits, size of encoding) into the dictionary
        self.data. At last, it updates object specific parameters.

        Returns:
            (None)
                Modifies self.data and self.allocated_qubits
        """
        for generator in self.network.generators.index:
            self.create_qubit_entries_for_component(
                component_name=generator,
                snapshot_to_weight_dict={
                    time : self.generator_to_qubits(int(self.get_nominal_power(generator, time)))
                    for time in self.network.snapshots
                }
            )
        return

    def store_lines(self) -> None:
        """
        Assigns a number of qubits, according to the option set in
        self.config, to each line in self.network. For each line, line
        specific parameters (i.e. power, corresponding qubits, size of
        encoding) are as well written into the dictionary self.data. At
        last, it updates object specific parameters.
        
        Returns:
            (None)
                Modifies self.data and self.allocated_qubits
        """
        for line in self.network.lines.index:
            # we assume that the capacity of a line is constant across all
            # snapshots
            constant_line_capacity = self.line_to_qubits(int(self.network.lines.loc[line].s_nom))
            weight_dict = {
                time : constant_line_capacity 
                for time in self.network.snapshots
            }
            self.create_qubit_entries_for_component(
                component_name=line,
                snapshot_to_weight_dict=weight_dict
            )


    def create_qubit_entries_for_component(self,
                                           component_name: str,
                                           snapshot_to_weight_dict: dict
                                           ) -> None:
        """
        A function to create qubits in the self.data dictionary that
        represent some network components. The qubits can be accessed
        using the component_name.
        The method places several restriction on what it accepts in
        order to generate a valid QUBO later on. The checks are intended
        to prevent name or qubit collision.

        Allocating qubits to a component has two effects on the `self.data`
        attribute, which stores the relation between components and qubits.
        It creates a dictionary which has the snapshots of the network as keys
        with the corresponding values being a list of integers, that contain
        all qubits that represent the given qubit at that point in time.
        This dictionary is stored in `self.data` with the key component name
        as the key.
        For each qubit(which are labeled using integers) that has been 
        allocated to represent a network component, it also stored it's weight
        at the top level of the `self.data` attribute

        Args:
            component_name: (str)
                The string used to couple the component with qubits.
            snapshot_to_weight_dict: (dict)
                a dictionary of lists with snapshots as keys and the 
                corresponding value being the weights of the qubits 
                representing the component at that snapshot

        Returns:
            (None)
                Modifies self.data and self.allocated_qubits
        """
        if isinstance(component_name, int):
            raise ValueError("Component names mustn't be of type int")
        if component_name in self._qubit_encoding:
            raise ValueError("Component name has already been used")

        # fill snapshot to weight relation with empty lists for snapshots
        # that implicitly require 0 qubits since they don't appear in the 
        # snapshot_to_weight_dict dictionary
        snapshot_to_weight_dict = {snapshot : snapshot_to_weight_dict.get(snapshot, [])
                for snapshot in self.network.snapshots}

        #store component - snapshot - representing qubit relation
        self._qubit_encoding[component_name] = {
            snapshot : self.allocate_qubits_to_weight_list(snapshot_to_weight_dict[snapshot])
            for snapshot in self.network.snapshots
        }
        # expose qubit weights at top level of `self.data`
        for snapshot, qubit_list in self._qubit_encoding[component_name].items():
            for idx, qubit in enumerate(qubit_list):
                self._qubit_weights[qubit] = snapshot_to_weight_dict[snapshot][idx]


    def allocate_qubits_to_weight_list(self, weight_list: list):
        """
        For a given list of weights, returns a list of qubits which will we mapped
        to these weights, starting at the quibt self.allocated_qubits and increasing
        that counter appropriately

        Args:
            weight_list: (list[float]) a list of floats, which describes weights to be
                mapped to newly allocated qubits
        Returns:
            (list[int]) a list of consecutive integers, which represent qubits which are
                    mapped to the weight list. Also increases internal qubit count
        """
        num_new_allocated_qubits = len(weight_list)
        allocated_qubit_list = list(range(self.allocated_qubits, 
                                            self.allocated_qubits + num_new_allocated_qubits))
        self.allocated_qubits += num_new_allocated_qubits
        return allocated_qubit_list


    # helper functions to set encoded values
    def set_output_network(self, solution: list) -> pypsa.Network:
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
        output_network = self.network.copy()
        # get Generator/Line Status
        for time in self.snapshots:
            for generator in output_network.generators.index:
                # set value in status-dataframe in generators_t dictionary
                status = int(
                    self.get_generator_status(gen=generator,
                                              solution=solution,
                                              time=time))
                column_status = list(output_network.generators_t.status.columns)
                if generator in column_status:
                    index_generator = column_status.index(generator)
                    output_network.generators_t.status.iloc[
                        time, index_generator] = status
                else:
                    output_network.generators_t.status[generator] = status

                # set value in p-dataframe in generators_t dictionary
                p = self.get_encoded_value_of_component(component=generator,
                                                        solution=solution,
                                                        time=time)
                columns_p = list(output_network.generators_t.p.columns)
                if generator in columns_p:
                    index_generator = columns_p.index(generator)
                    output_network.generators_t.p.iloc[
                        time, index_generator] = p
                else:
                    output_network.generators_t.p[generator] = p

                # set value in p_max_pu-dataframe in generators_t dictionary
                columns_p_max_pu = list(
                    output_network.generators_t.p_max_pu.columns)
                p_nom = output_network.generators.loc[generator, "p_nom"]
                if p == 0:
                    p_max_pu = 0.0
                else:
                    p_max_pu = p_nom / p
                if generator in columns_p_max_pu:
                    index_generator = columns_p_max_pu.index(generator)
                    output_network.generators_t.p_max_pu.iloc[
                        time, index_generator] = p_max_pu
                else:
                    output_network.generators_t.p_max_pu[generator] = p_max_pu

            for line in output_network.lines.index:
                encoded_val = self.get_encoded_value_of_component(
                    component=line,
                    solution=solution,
                    time=time)
                # p0 - Active power at bus0 (positive if branch is withdrawing
                # power from bus0).
                # p1 - Active power at bus1 (positive if branch is withdrawing
                # power from bus1).
                p0 = encoded_val
                p1 = -encoded_val

                columns_p0 = list(output_network.lines_t.p0.columns)
                if line in columns_p0:
                    index_line = columns_p0.index(line)
                    output_network.lines_t.p0.loc[time, index_line] = p0
                else:
                    output_network.lines_t.p0[line] = p0

                columns_p1 = list(output_network.lines_t.p1.columns)
                if line in columns_p1:
                    index_line = columns_p1.index(line)
                    output_network.lines_t.p1.loc[time, index_line] = p1
                else:
                    output_network.lines_t.p1[line] = p1

        return output_network

    # helper functions for getting encoded values
    def get_qubit_encoding(self) -> dict:
        """
        Returns the dictionary that holds information on the encoding
        of the network into qubits.
        
        Returns:
            (dict)
                The dictionary with network component as keys and qubit
                information as values
        """
        return self._qubit_encoding

    def get_bus_components(self, bus: str) -> dict:
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
                'positive_lines':    A list of labels of lines that end
                                    in this bus.
                'negative_lines':    A list of labels of lines that start
                                    in this bus.
        """
        if bus not in self.network.buses.index:
            raise ValueError("the bus " + bus + " doesn't exist")
        result = {
            "generators":
                list(self.network.generators[
                         self.network.generators.bus == bus
                         ].index),
            "positive_lines":
                list(self.network.lines[
                         self.network.lines.bus1 == bus
                         ].index),
            "negative_lines":
                list(self.network.lines[
                         self.network.lines.bus0 == bus
                         ].index),
        }
        return result

    def get_nominal_power(self, generator: str, time: any) -> float:
        """
        Returns the nominal power of a generator at a time step saved
        in the network.
        
        Args:
            generator: (str)
                The generator label.
            time: (int)
                Index of time slice for which to get nominal power. Has to
                be in the index self.network.snapshots

        Returns:
            (float)
                Nominal power available at 'generator' at time slice
                'time'
        """
        try:
            p_max_pu = self.network.generators_t.p_max_pu[generator].loc[time]
        except KeyError:
            p_max_pu = 1.0
        return max(self.network.generators.p_nom[generator] * p_max_pu, 0)

    def get_generator_status(self, gen: str, solution: list, time: any 
                             ) -> bool:
        """
        Returns the status of a generator 'gen' (i.e. on or off) at a
        time slice 'time' in a given solution. If the generator is
        represented by multiple qubits, the first qubit of that snapshot
        is assumed to model the status of the generator

        Args:
            gen: (str)
                The label of the generator.
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            time: (int)
                Index of time slice for which to get the generator
                status. This has to be in the network.snapshots index
        """
        try:
            return self._qubit_encoding[gen][time][0] in solution
        except IndexError:
            return False

    def get_generator_dictionary(self, solution: list,
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
            for time in self.snapshots:
                key = (generator, time)
                if stringify:
                    key = str(key)
                result[key] = int(
                    self.get_generator_status(
                            gen=generator, 
                            solution=solution,
                            time=time)
                            )
        return result

    def get_flow_dictionary(self, solution: list,
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
            for time in self.snapshots:
                key = (lineId, time)
                if stringify:
                    key = str(key)
                result[key] = self.get_encoded_value_of_component(
                    component=lineId, solution=solution, time=time)
        return result

    def get_load(self, bus: str, time: any, silent: bool = True) -> float:
        """
        Returns the total load at a bus at a given time slice.

        Args:
            bus: (str)
                Label of bus at which to calculate the total load.
            time: (int)
                Index of time slice for which to get the total load.
            silent: (bool)
                Flag for turning print on or off

        Returns:
            (float)
                The total load at 'bus' at time slice 'time'.
        """
        loads_at_current_bus = self.network.loads[
            self.network.loads.bus == bus
            ].index
        all_loads = self.network.loads_t['p_set'].loc[time]
        result = all_loads[all_loads.index.isin(loads_at_current_bus)].sum()
        if result == 0:
            if not silent:
                print(f"Warning: No load at {bus} at timestep {time}.\n"
                      f"Falling back to constant load")
            all_loads = self.network.loads['p_set']
            result = all_loads[all_loads.index.isin(loads_at_current_bus)].sum()
        if result < 0:
            raise ValueError("negative Load at current Bus")
        return result

    def get_total_load(self, time: any) -> float:
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
            load += self.get_load(bus, time)
        return load

    def get_representing_qubits(self, component: str, time: any = None) -> list:
        """
        Returns a list of all qubits that are used to encode a network
        component at a given time slice.

        Args:
            component: (str)
                Label of the (network) component.
            time: (int)
                Index of time slice for which to get representing
                qubits. This has to be in the network.snapshots index

        Returns:
            (list)
                List of integers which are qubits that represent the
                component.
        """
        if time is None:
            time = self.network.snapshots[0]
        return self._qubit_encoding[component][time]

    def get_qubit_mapping(self, time: any = None) -> dict:
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
        return {component: self.get_representing_qubits(component=component,
                                                        time=time)
                for component in self._qubit_encoding.keys()
                if isinstance(component, str)}

    def get_interaction(self, *args) -> float:
        """
        Returns the interaction coefficient of a list of qubits.
        
        Args:
            args: (int)
                All qubits that are involved in this interaction.

        Returns:
            (float)
                The interaction strength between all qubits in args.
        """
        sorted_unique_arguments = tuple(sorted(set(args)))
        return self.ising_coefficients.get(sorted_unique_arguments, 0.0)

    def get_encoded_value_of_component(self, component: str, solution: list,
                                       time: any = 0) -> float:
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
            time: (any)
                Index of time slice for which to retrieve the encoded value.
                It has to be in the index self.network.snapshots

        Returns:
            (float)
                Value of the component encoded in the spin configuration
                of solution.
        """
        value = 0.0
        for qubit in self._qubit_encoding[component][time]:
            if qubit in solution:
                value += self._qubit_weights[qubit]
        return value

    def generate_report(self, solution: list) -> dict:
        """
        For the given solution, calculates various properties of the
        solution.

        Args:
            solution: (list)
                List of all qubits that have spin -1 in a solution.
        Returns:
            (dict)
                A dictionary containing general information about the
                solution.
        """
        return {
            "total_cost": self.calc_cost(solution=solution),
            "kirchhoff_cost": self.calc_kirchhoff_cost(solution=solution),
            "kirchhoff_cost_by_time": self.calc_kirchhoff_cost_by_time(solution=solution),
            "power_imbalance": self.calc_power_imbalance(solution=solution),
            "total_power": self.calc_total_power_generated(solution=solution),
            "marginal_cost": self.calc_marginal_cost(solution=solution),
            "individual_kirchhoff_cost": self.individual_cost_contribution(
                solution=solution),
            "unit_commitment": self.get_generator_dictionary(solution=solution,
                                                            stringify=True),
            "powerflow": self.get_flow_dictionary(solution=solution,
                                                  stringify=True),
        }

    def calc_cost(self, solution: list,
                  ising_interactions: dict = None) -> float:
        """
        Calculates the energy of a spin state including the constant
        energy contribution.
        The default Ising spin glass state that is used to calculate
        the energy of a solution is the full problem stored in the
        IsingBackbone. Ising subproblems can overwrite which Ising
        interactions are used to calculate the energy to get subproblem
        specific information. The assignment of qubits to the network is
        still fixed.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            ising_interactions: (dict)
                The Ising problem to be used to calculate the energy.

        Returns:
            (float)
                The energy of the spin glass state in solution.
        """
        solution = set(solution)
        if ising_interactions is None:
            ising_interactions = self.ising_coefficients
        total_cost = 0.0
        for spins, weight in ising_interactions.items():
            if len(spins) == 1:
                factor = 1
            else:
                factor = -1
            for spin in spins:
                if spin in solution:
                    factor *= -1
            total_cost += factor * weight
        return total_cost

    def individual_marginal_cost(self, solution: list) -> dict:
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
                       **self.calc_marginal_cost_at_bus(bus=bus,
                                                        solution=solution)
                       }
        return contrib


    def calc_marginal_cost_at_bus(self, bus: str, solution: list) -> dict:
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
        for time in self.snapshots:
            marginal_cost = 0.0
            components = self.get_bus_components(bus)
            for generator in components['generators']:
                power_output_at_current_time = self.get_encoded_value_of_component(
                    component=generator,
                    solution=solution,
                    time=time
                )
                marginal_cost += power_output_at_current_time * \
                                self.network.generators["marginal_cost"].loc[
                                     generator
                                 ] 
            contrib[str((bus, time))] = marginal_cost
        return contrib

    def calc_marginal_cost(self, solution: list) -> float:
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
        marginal_cost = 0.0
        for key, val in self.individual_marginal_cost(solution=solution).items():
            marginal_cost += val
        return marginal_cost

    # getter for encoded ising problem parameters
    def siquan_format(self) -> list:
        """
        Returns the complete problem in the format required by the
        siquan solver.

        Returns:
            (list)
                A list of tuples of the form (interaction-coefficient,
                list(qubits)).
        """
        return [(v, list(k)) for k, v in self.ising_coefficients.items() if
                v != 0 and len(k) > 0]

    def get_hamiltonian_matrix(self) -> list:
        """
        Returns a matrix containing the Ising hamiltonian

        Returns:
            (list)
                A list of lists representing the hamiltonian matrix.
        """
        qubits = range(self.allocated_qubits)
        hamiltonian = [
            [self.get_interaction(i, j) for i in qubits] for j in qubits
        ]
        return hamiltonian

    def get_hamiltonian_eigenvalues(self) -> np.ndarray:
        """
        Returns the eigenvalues and normalized eigenvectors of the
        hamiltonian matrix.

        Returns:
            (np.ndarray)
                A numpy array containing all eigenvalues.
        """
        return np.linalg.eigh(self.get_hamiltonian_matrix())


class AbstractIsingSubproblem:
    """
    An interface for classes that model the Ising formulation
    subproblem of an unit commitment problem.
    Classes that model a subproblem/constraint are subclasses of this
    class and adhere to the following structure. Each subproblem/
    constraint corresponds to one class. This class has a factory method
    which chooses the correct subclass of itself. Any of those have a
    method that accepts an IsingBackbone as the argument and encodes
    its problem onto that object.
    """

    def __init__(self, backbone: IsingBackbone, config: dict):
        """
        The constructor for a subproblem to be encoded into the Ising
        subproblem. Different formulations of the same subproblems use
        a (factory) classmethod to choose the correct subclass and call this
        constructor. The attributes set here are the minimal attributes
        that are expected. The attributes we set have the following purpose:
            ising_coefficients: (dict) this contains the qubo formulation of just the
                subproblem
            scale_factor: (float) this contains a linear factor to scale the
                problem with 
            backbone: (IsingBackbone) This is the IsingBackbone instance that
                the AbstractIsingSubproblem instance uses to formulate the
                constraint
            network: (pypsa.Network) The underlying network of the unit commitment
                problem
        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance.
        """
        self.ising_coefficients = {}
        try:
            self.scale_factor = config["scale_factor"]
        except KeyError:
            print("Can't find value for 'scale_factor', fallback to '1.0'")
            self.scale_factor = 1.0
        self.backbone = backbone
        self.network = backbone.network

    @classmethod
    def build_subproblem(cls, backbone: IsingBackbone,
                         configuration: dict) -> 'AbstractIsingSubproblem':
        """
        Returns an instance of the class set up according to the
        configuration. This is done by choosing the corresponding subclass
        of the configuration. After initialization, the instance can encode
        this subproblem into the ising_backbone by calling the encode_subproblem 
        method.

        Args:
            backbone: (IsingBackbone)
                The ising_backbone used to encode the subproblem.
            configuration: (dict)
                The configuration dictionary, containing all data
                necessary to initialize.

        Returns:
            (AbstractIsingSubproblem)
                The constructed Ising subproblem.
        """
        raise NotImplementedError

    def encode_subproblem(self):
        """
        This encodes the problem an instance of a subclass is describing
        into the ising_backbone instance. After this call, the
        corresponding QUBO is stored in the ising_backbone.
        """
        raise NotImplementedError

    def calc_cost(self, solution: list) -> float:
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
        return self.backbone.calc_cost(solution=solution,
                                       ising_interactions=self.ising_coefficients)


class MarginalCostSubproblem(AbstractIsingSubproblem, ABC):
    """
    An interface for classes that model the Marginal Cost subproblem of
    an unit commitment problem.
    Classes that model a marginal cost subproblem are subclasses of this
    class and adhere to the structure inherited from
    'AbstractIsingSubproblem'.
    """

    @classmethod
    def build_subproblem(cls, backbone: IsingBackbone,
                         configuration: dict) -> 'MarginalCostSubproblem':
        """
        A factory method for obtaining the marginal cost model specified
        in configuration.
        Returns an instance of the class set up according to the
        configuration.
        This is done by choosing the corresponding subclass of the
        configuration. After initialization, the instance can encode
        this subproblem into the ising_backbone by calling the
        encode_subproblem method.

        Args:
            backbone: (IsingBackbone)
                The ising_backbone used to encode the subproblem.
            configuration: (dict)
                The configuration dictionary, containing all data
                necessary to initialize.

        Returns:
            (MarginalCostSubproblem)
                The constructed Ising instance ready to encode into the
                backbone.
        """
        subclass_table = {
            "global_cost_square": GlobalCostSquare,
            "global_cost_square_with_slack": GlobalCostSquareWithSlack,
            "marginal_as_penalty": MarginalAsPenalty,
            "local_marginal_estimation": LocalMarginalEstimation,
        }
        return subclass_table[configuration.setdefault("strategy", "global_cost_square")](
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
        constructor. Additionally, it sets three more parameters which
        slightly change how the penalty is applied:
            `offset_estimation_factor`: sets an offset across all
                generators by the cost of the most efficient generator
                scaled by this factor

        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance.
        """
        super().__init__(backbone, config)
        # factor which in conjunction with the minimal cost per energy produced
        # describes by how much each marginal cost per unit produced is offset
        # in all generators to bring the average cost of a generator closer to
        # zero
        self.offset_estimation_factor = float(config["offset_estimation_factor"])

    def encode_subproblem(self) -> None:
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
        for time in self.network.snapshots:
            for bus in self.network.buses.index:
                self.encode_marginal_costs(bus=bus, time=time)

    def marginal_cost_offset(self) -> float:
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
                The offset of all generator's marginal costs.
        """
        return 1.0 * min(self.network.generators[
                             "marginal_cost"]) * self.offset_estimation_factor

    def encode_marginal_costs(self, bus: str, time: int) -> None:
        """
        Encodes marginal costs for running generators and transmission
        lines at a single bus.
        This uses an offset calculated in ´marginal_cost_offset´, which is
        dependent on all generators of the entire network for a single
        time slice.

        Args:
            bus: (str)
                Label of the bus at which to add marginal costs.
            time: (int)
                Index of time slice for which to add marginal cost.

        Returns:
            (None)
                Modifies self.ising_coefficients. Adds to previously written
                interaction coefficient.
        """
        generators = self.backbone.get_bus_components(bus)['generators']
        cost_offset = self.marginal_cost_offset()
        for generator in generators:
            self.backbone.couple_component_with_constant(
                component=generator,
                coupling_strength=self.scale_factor
                                  * (self.network.generators["marginal_cost"].loc[generator]
                                     - cost_offset),
                time=time
            )


class LocalMarginalEstimation(MarginalCostSubproblem):
    """
    A subproblem class that models the minimization of the marginal
    costs. It does this by estimating the cost at each bus and models a
    minimization of the distance of the incurred cost to the estimated
    cost.
    """

    def __init__(self, backbone: IsingBackbone, config: dict):
        """
        The constructor for encoding marginal cost as distance between
        the incurred and estimated costs.
        It inherits its functionality from the AbstractIsingSubproblem
        constructor. Additionally, it sets three more parameters which
        slightly change how the penalty is applied:
            `offset_estimation_factor`: sets an offset across all
                generators by the cost of the most efficient generator
                scaled by this factor

        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance.
        """
        super().__init__(backbone, config)
        self.offset_estimation_factor = float(config["offset_estimation_factor"])

    def encode_subproblem(self) -> None:
        # TODO: check DocString
        """
        Encodes the minimization of the marginal cost by estimating the
        cost at each bus and modeling a minimization of the distance of
        the incurred cost to the estimated cost. The exact modeling of
        this can be adjusted using the parameters: ''

        Returns:
            (None)
                Modifies self.backbone.
        """
        # Estimation is done independently at each bus. Thus, it suffices to
        # iterate over all snapshots and buses to encode the subproblem
        for time in self.network.snapshots:
            for bus in self.network.buses.index:
                self.encode_marginal_costs(bus=bus, time=time)

    def choose_offset(self, sorted_generators: list) -> float:
        """
        Calculates a float by which to offset all marginal costs. The
        chosen offset is the minimal marginal cost of the generators in
        the provided list.

        Args:
            sorted_generators: (list)
                A list of generators already sorted by their minimal
                cost in ascending order.

        Returns:
            (float)
                The offset by which to adjust all marginal costs of
                network components.
        """
        # there are lots of ways to choose an offset. offsetting such that 0 is
        # minimal cost is decent but for example choosing an offset slightly
        # over that seems to also produce good results. It is not clear how
        # important the same sign on all marginal costs is
        marginal_cost_list = [self.network.generators["marginal_cost"].loc[gen]
                              for gen in sorted_generators]
        return self.offset_estimation_factor * np.min(marginal_cost_list)

    def estimate_marginal_cost_at_bus(self,
                                      bus: str,
                                      time: int
                                      ) -> [float, float]:
        """
        Estimates a lower bound for marginal costs incurred by matching
        the load at the bus only with generators that are at this bus.

        Args:
            bus: (str)
                Label of the bus at which to estimate marginal costs.
            time: (int)
                Index of the time slice for which to estimate marginal
                costs.

        Returns:
            (float)
                An estimation of the incurred marginal cost if the
                marginal costs of generators are all offset by the
                second return value.
            (float)
                The value by which to offset the marginal costs of all
                generators.
        """
        remaining_load = self.backbone.get_load(bus, time)
        generators = self.backbone.get_bus_components(bus)['generators']
        sorted_generators = sorted(
            generators,
            key=lambda gen: self.network.generators["marginal_cost"].loc[gen]
        )
        offset = self.choose_offset(sorted_generators)
        cost_estimation = 0.0
        for generator in sorted_generators:
            if remaining_load <= 0:
                break
            supplied_power = min(self.backbone.get_nominal_power(
                                            generator=generator, 
                                            time=time),
                                load)
            cost_estimation += supplied_power * (
                    self.network.generators["marginal_cost"].loc[
                        generator] - offset)
            remaining_load -= supplied_power
        return cost_estimation, offset

    def calculate_cost(self,
                       component_to_be_valued: str,
                       all_components: dict,
                       offset: float,
                       estimated_cost: float,
                       load: float
                       ) -> float:
        # TODO: check DocString
        """
        Calculates and returns the marginal costs of a component
        'componentToBeValued'.

        Args:
            component_to_be_valued: (str)
                Label of the component at which to estimate marginal
                costs.
            all_components: (dict)
                All components connect to a specific bus, including
                'componentToBeValued'.
            offset: (float)
                The value by which to offset the marginal costs of all
                generators.
            estimated_cost: (float)
                An estimation of the incurred marginal cost.
            load: (float)
                The load on the bus, to which all given components are
                connected.

        Returns:
            (float)
                The marginal costs of the specified component.
        """
        if component_to_be_valued in all_components["generators"]:
            return self.network.generators["marginal_cost"].loc[
                       component_to_be_valued] - offset
        if component_to_be_valued in all_components["positive_lines"]:
            return 0.5 * estimated_cost / load
        if component_to_be_valued in all_components["negative_lines"]:
            return 0.5 * - estimated_cost / load

    def encode_marginal_costs(self, bus: str, time: any) -> None:
        """
        Encodes marginal costs at a bus by first estimating a lower
        bound of unavoidable marginal costs, then deviations in the
        marginal cost from that estimation are penalized quadratically.

        Args:
            bus: (str)
                Label of the bus at which to encode marginal costs.
            time: (int)
                Index of the time slice for which to estimate marginal
                costs.

        Returns:
            (None)
                Modifies self.backbone. Adds to previously written
                interaction coefficient
        """
        components = self.backbone.get_bus_components(bus)
        flattened_componenents = components['generators'] + \
                                 components['positive_lines'] + \
                                 components['negative_lines']

        estimated_cost, offset = self.estimate_marginal_cost_at_bus(bus, time)
        load = self.backbone.get_load(bus, time)

        self.backbone.add_interaction(0.25 * estimated_cost ** 2)
        for first_component in flattened_componenents:
            self.backbone.couple_component_with_constant(
                first_component,
                - 2.0 * self.calculate_cost(first_component,
                                            components,
                                            offset,
                                            estimated_cost,
                                            load,
                                            )
                * estimated_cost
                * self.scale_factor,
                time=time
            )
            for second_component in flattened_componenents:
                current_factor = self.scale_factor * \
                                 self.calculate_cost(
                                     first_component,
                                     components,
                                     offset,
                                     estimated_cost,
                                     load,
                                 ) * \
                                 self.calculate_cost(
                                     second_component,
                                     components,
                                     offset,
                                     estimated_cost,
                                     load,
                                 )
                self.backbone.couple_components(
                    first_component,
                    second_component,
                    coupling_strength=current_factor
                )


class GlobalCostSquare(MarginalCostSubproblem):
    """
    A subproblem class that models the minimization of the marginal
    costs. It does this by estimating it and then modelling the squared
    distance of the actual cost to the estimated cost.
    """

    def __init__(self, backbone: IsingBackbone, config: dict):
        """
        A constructor for encoding marginal cost as quadratic penalties.
        It inherits its functionality from the AbstractIsingSubproblem
        constructor. Additionally, it sets three more parameters which
        slightly change how the penalty is applied:
            `offset_estimation_factor`: sets an offset across all
                generators by the cost of the most efficient generator
                scaled by this factor

        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance.
        """
        super().__init__(backbone, config)
        self.offset_estimation_factor = float(config.setdefault("offset_estimation_factor", 1.0))
        config

    def print_estimation_report(self,
                                estimated_cost: float,
                                offset: float,
                                time: int
                                ) -> None:
        """
        Prints the estimated marginal cost and the offset of the cost
        per MW produced.
    
        Args:
            estimated_cost: (float)
                An estimation of the incurred marginal cost.
            offset: (float)
                The value by which to offset the marginal costs of all
                generators.
            time: (int)
                Index of the time slice for which to print the estimated
                marginal costs and offset.
        Returns:
            (None)
                Prints to stdout.
        """
        print()
        print(f"--- Estimation Parameters at timestep {time} ---")
        print(f"Absolute offset: {offset}")
        print(f"Lower limit of minimal cost (with offset): {estimated_cost}")
        print(
            f"Current total estimation at {time}:"
            f" {offset * self.backbone.get_total_load(time)}")
        print("---")

    def encode_subproblem(self) -> None:
        """
        Encodes the square of the marginal cost onto the energy. This
        results in a minimization of the marginal costs.

        Returns:
            (None)
                Modifies self.backbone.
        """
        for time in self.network.snapshots:
            self.encode_marginal_costs(time=time)

    def choose_offset(self, sorted_generators: list) -> float:
        """
        Calculates the offset, by which to offset all marginal costs.
        The chosen offset is the minimal marginal cost of a generator in
        'sorted_generators'.

        Args:
            sorted_generators: (list)
                A list of generators already sorted by their minimal
                cost in ascending order.

        Returns:
            (float)
                The value, by which to offset all marginal costs of the
                network components.
        """
        # there are lots of ways to choose an offset. offsetting such that 0 is
        # minimal cost is decent but for example choosing an offset slightly
        # over that seems to also produce good results. It is not clear how
        # important the same sign on all marginal costs is
        marginal_cost_list = [self.network.generators["marginal_cost"].loc[gen]
                              for gen in sorted_generators]
        return self.offset_estimation_factor * np.min(marginal_cost_list)

    def estimate_global_marginal_cost(self,
                                      time: int,
                                      expected_additonal_cost: float = 0.0
                                      ) -> [float, float]:
        """
        Estimates a lower bound of incurred marginal costs if locality
        of generators could be ignored at a given time slice.
        Unavoidable baseline costs of matching the load is ignored. The
        offset to reduce baseline costs to 0 and estimated marginal cost
        with a constant is returned.

        Args:
            time: (int)
                Index of time slice for which to calculate the lower
                bound of offset marginal cost.
            expected_additonal_cost: (float)
                Constant by which to offset the returned marginal cost.
                Default: 0.0

        Returns:
            (float)
                An estimation of the incurred marginal cost if the
                marginal costs of generators are all offset by the
                second return value.
            (float)
                The value by which to offset the marginal costs of all
                generators.
        """
        load = 0.0
        for bus in self.network.buses.index:
            load += self.backbone.get_load(bus, time)

        sorted_generators = sorted(
            self.network.generators.index,
            key=lambda gen: self.network.generators["marginal_cost"].loc[gen]
        )
        offset = self.choose_offset(sorted_generators)
        cost_estimation = 0.0
        for generator in sorted_generators:
            if load <= 0:
                break
            supplied_power = min(self.backbone.get_nominal_power(
                                            generator=generator, 
                                            time=time),
                                load)
            cost_estimation += supplied_power * (
                    self.network.generators["marginal_cost"].loc[
                        generator] - offset)
            load -= supplied_power
        return cost_estimation + expected_additonal_cost, offset

    # TODO refactor using a isingbackbone function for encoding squared distances
    def encode_marginal_costs(self, time: int) -> None:
        """
        The marginal costs of using generators are considered one single
        global constraint. The square of marginal costs is encoded
        into the energy and thus minimized.

        Args:
            time: (int)
                Index of time slice for which to encode marginal costs.

        Returns:
            (None)
                Modifies self.ising_coefficients. Adds to previously written
                interaction coefficient.
        """
        # TODO make this more readable
        estimated_cost, offset = \
            self.estimate_global_marginal_cost(time=time,
                                               expected_additonal_cost=0)
        self.print_estimation_report(estimated_cost=estimated_cost,
                                     offset=offset,
                                     time=time)
        generators = self.network.generators.index
        # estimation of marginal costs is a global estimation.
        # Calculate total power needed
        # offset the marginal costs per energy produces and encode problem
        # into backbone
        for first_generator in generators:
            marginal_cost_first_generator = self.network.generators["marginal_cost"].loc[
                                                first_generator] - offset
            for second_generator in generators:
                marginal_cost_second_generator = \
                    self.network.generators["marginal_cost"].loc[second_generator] - offset
                current_factor = self.scale_factor * \
                                 marginal_cost_first_generator * \
                                 marginal_cost_second_generator
                self.backbone.couple_components(
                    first_component=first_generator,
                    second_component=second_generator,
                    coupling_strength=current_factor
                )


class KirchhoffSubproblem(AbstractIsingSubproblem):
    """
    A class that models the Kirchhoff subproblem of an unit commitment
    problem.
    """

    def __init__(self, backbone: IsingBackbone, config: dict):
        """
        A constructor for encoding the kirchhoff subproblem onto the
        provided IsingBackbone.
        It inherits its functionality from the AbstractIsingSubproblem
        constructor.

        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance.
        """
        super().__init__(backbone=backbone, config=config)

    @classmethod
    def build_subproblem(cls,
                         backbone: IsingBackbone,
                         configuration: dict
                         ) -> 'KirchhoffSubproblem':
        """
        A factory method for initializing the kirchhoff subproblem
        model.
        Returns an instance of the KirchhoffSubproblem class. This can
        then be encoded onto the ising_backbone by calling the
        encode_subproblem method.

        Args:
            backbone: (IsingBackbone)
                The ising_backbone used to encode the subproblem.
            configuration: (dict)
                The configuration dictionary, containing all data
                necessary to initialize.

        Returns:
            (KirchhoffSubproblem)
                The constructed Ising instance ready to encode into the
                backbone.
        """
        return KirchhoffSubproblem(backbone, configuration)

    def encode_subproblem(self) -> None:
        """
        Encodes the kirchhoff constraint at each bus.

        Returns:
            (None)
                Modifies self.backbone.
        """
        for time in self.network.snapshots:
            for bus in self.network.buses.index:
                self.encode_kirchhoff_constraint(ising_backbone=self.backbone,
                                                 bus=bus,
                                                 time=time)
        self.ising_coefficients = self.backbone.cached_problem

    def encode_kirchhoff_constraint(self,
                                    ising_backbone: IsingBackbone,
                                    bus: str,
                                    time: any 
                                    ) -> None:
        """
        Adds the kirchhoff constraint at a bus to the problem
        formulation.
        The kirchhoff constraint is that the sum of all power generating
        elements (i.e. generators, power flow towards the bus) is equal
        to the sum of all load generating elements (i.e. bus specific
        load, power flow away from the bus).
        Deviation from equality is penalized quadratically.
        At a bus, the total power can be calculated as:
        (-Load + activeGenerators + powerflowTowardsBus
        - powerflowAwayFromBus) ** 2
        The function expands this expression and adds all result
        products of two components by looping over them.

        Args:
            ising_backbone: (IsingBackbone)
                Ising backbone onto which the kirchhoff subproblem
                is to be encoded.
            bus: (str)
                Label of the bus at which to enforce the kirchhoff
                constraint.
            time: (int)
                Index of time slice at which to enforce the kirchhoff
                constraint.

        Returns:
            (None)
                Modifies self.ising_coefficients. Adds to previously written
                interaction coefficient.
        """
        components = ising_backbone.get_bus_components(bus)
        flattened_components = components['generators'] + \
                               components['positive_lines'] + \
                               components['negative_lines']
        demand = ising_backbone.get_load(bus, time=time)

        # constant load contribution to cost function so that a configuration
        # that fulfills the kirchhoff constraint has energy 0
        ising_backbone.add_interaction(self.scale_factor * demand ** 2)
        for first_component in flattened_components:
            # this factor sets the scale, as well as the sign to encode if
            # an active component acts a generator or a load
            factor = self.scale_factor
            if first_component in components['negative_lines']:
                factor *= -1.0
            # reward/penalty term for matching/adding load. Contains all
            # products with the Load
            ising_backbone.couple_component_with_constant(first_component,
                                                          - 2.0 * factor * demand,
                                                        time=time)
            for second_component in flattened_components:
                # adjust sing for direction of flow at line
                if second_component in components['negative_lines']:
                    current_factor = -factor
                else:
                    current_factor = factor
                # attraction/repulsion term for different/same sign of power
                # at components
                ising_backbone.couple_components(first_component,
                                                second_component,
                                                coupling_strength=current_factor,
                                                time=time)


    def calc_power_imbalance_at_bus_at_time(self,
                                    bus: str,
                                    time: any,
                                    result: list,
                                    ) -> dict:
        """
        Returns a dictionary containing the absolute values of the power
        imbalance/mismatch at a bus for one particular time step
        
        Args:
            bus: (str)
                Label of the bus at which to calculate the power
                imbalances.
            time: (any)
                snapshot at which to calculate the power imbalance
            result: (list)
                List of all qubits which have spin -1 in the solution

        Returns:
            (dict)
                Dictionary containing the power imbalance of 'bus' at
                all time slices. The keys are tuples of the label of the
                bus and the index of the time slice. The values are
                floats, representing the power imbalance of the bus at
                the time slice.
        """
        load = - self.backbone.get_load(bus, time)
        components = self.backbone.get_bus_components(bus)
        for gen in components['generators']:
            load += self.backbone.get_encoded_value_of_component(gen, result,
                                                                 time=time)
        for line_id in components['positive_lines']:
            load += self.backbone.get_encoded_value_of_component(line_id,
                                                                 result,
                                                                 time=time)
        for line_id in components['negative_lines']:
            load -= self.backbone.get_encoded_value_of_component(line_id,
                                                                 result,
                                                                 time=time)
        return load


    def calc_power_imbalance_at_bus(self,
                                    bus: str,
                                    result: list,
                                    silent: bool = True
                                    ) -> dict:
        """
        Returns a dictionary containing the absolute values of the power
        imbalance/mismatch at a bus for each time step.
        
        Args:
            bus: (str)
                Label of the bus at which to calculate the power
                imbalances.
            result: (list)
                List of all qubits which have spin -1 in the solution
            silent: (bool)
                Switch to enable status messages send to stdout. If
                true, no messages are sent.
                Default: True

        Returns:
            (dict)
                Dictionary containing the power imbalance of 'bus' at
                all time slices. The keys are tuples of the label of the
                bus and the index of the time slice. The values are
                floats, representing the power imbalance of the bus at
                the time slice.
        """
        contrib = {}
        for time in self.network.snapshots:
            contrib[str((bus, time))] = self.calc_power_imbalance_at_bus_at_time(bus, time, result)
        return contrib

    def calc_total_power_generated_at_bus(self,
                                          bus: str,
                                          solution: list,
                                          time: any
                                          ) -> float:
        """
        Calculates how much power is generated using generators at this
        'bus' at the time slice with index 'time'.
        Ignores any power flow or load.

        Args:
            bus: (str)
                Label of the bus at which to calculate the total power.
            solution: (list)
                List of all qubits that have spin -1 in a solution.
            time: (int)
                Index of the time slice at which to calculate the total
                power.
        Returns:
            (float)
                The total power generated without flow or loads.
        """
        total_power = 0.0
        generators = self.backbone.get_bus_components(bus=bus)['generators']
        for generator in generators:
            total_power += self.backbone.get_encoded_value_of_component(
                component=generator, solution=solution, time=time)
        return total_power

    def calc_total_power_generated(self,
                                   solution: list,
                                   ) -> float:
        """
        Calculates how much power is generated using generators across
        the entire network at a time slice with index 'time'.
    
        Args:
            solution: (list)
                List of all qubits that have spin -1 in a solution.
            time: (int)
                Index of the time slice at which to calculate the total
                power generated across the whole network.
        Returns:
            (float)
                The total power generated across the whole network
                without flow or loads.
        """
        total_power = 0.0
        for bus in self.network.buses.index:
            for time in self.network.snapshots:
                total_power += self.calc_total_power_generated_at_bus(bus=bus,
                                                                      solution=solution,
                                                                      time=time)
        return total_power

    def calc_power_imbalance(self, solution: list) -> float:
        """
        Returns the sum of all absolutes values of power imbalances at
        each bus over all time slices.
        This is basically like the kirchhoff cost except with a linear
        penalty.
        
        Args:
            solution: (list)
                List of all qubits which have spin -1 in the solution.
        Returns:
            (float)
                The sum of all absolute values of every power imbalance
                at every bus.
        """
        power_imbalance = 0.0
        for bus in self.network.buses.index:
            for _, imbalance in self.calc_power_imbalance_at_bus(
                    bus=bus, result=solution).items():
                power_imbalance += abs(imbalance)
        return power_imbalance

    def calc_kirchhoff_cost_at_bus(self,
                                   bus: str,
                                   result: list,
                                   silent: bool = True
                                   ) -> dict:
        """
        Returns a dictionary which contains the kirchhoff cost at the
        'bus' for every time slice with index 'time', scaled by the
        KirchhoffFactor.

        Args:
            bus: (str)
                Label of the bus at which to calculate the total power.
            result: (list)
                List of all qubits that have spin -1 in a solution.
            silent: (bool)
                Switch to enable status messages send to stdout. If
                true, no messages are sent.
                Default: True
        Returns:
            (dict)
                Dictionary containing the kirchhoff cost of 'bus' at
                all time slices. The keys are tuples of the label of the
                bus and the index of the time slice. The values are
                floats, representing the kirchhoff cost of the bus at
                the time slice.
        """
        return {
            key: (imbalance * self.scale_factor) ** 2
            for key, imbalance in
            self.calc_power_imbalance_at_bus(bus=bus,
                                             result=result,
                                             silent=silent).items()
        }


    def calc_kirchhoff_cost_by_time(self, solution: list) -> float:
        """
        Calculate the total unscaled kirchhoff cost incurred by a
        solution at each time step.
        
        Args:
            solution: (list)
                List of all qubits which have spin -1 in the solution.
            
        Returns:
            (dict)
                A dictionary with snapshots as keys and the kirchhoff
                cost at that time step as values
        """
        result = {}
        for time in self.network.snapshots:
            kirchhoff_cost = 0.0
            for bus in self.network.buses.index:
                kirchhoff_cost += self.calc_power_imbalance_at_bus_at_time(bus, time, solution) ** 2
            result[time] = kirchhoff_cost
        return result


    def calc_kirchhoff_cost(self, solution: list) -> float:
        """
        Calculate the total unscaled kirchhoff cost incurred by a
        solution.
        
        Args:
            solution: (list)
                List of all qubits which have spin -1 in the solution.
            
        Returns:
            (float)
                Total kirchhoff cost incurred without kirchhoffFactor
                scaling.
        """
        kirchhoff_cost = 0.0
        for bus in self.network.buses.index:
            for _, val in self.calc_power_imbalance_at_bus(
                    bus=bus, result=solution).items():
                kirchhoff_cost += val ** 2
        return kirchhoff_cost

    def individual_cost_contribution(self,
                                     solution: list,
                                     silent: bool = True
                                     ) -> dict:
        """
        Returns a dictionary which contains the kirchhoff costs incurred
        at all busses for every time slice, scaled by the
        KirchhoffFactor.

        Args:
            solution: (list)
                List of all qubits that have spin -1 in a solution.
            silent: (bool)
                Switch to enable status messages send to stdout. If
                true, no messages are sent.
                Default: True
        Returns:
            (dict)
                Dictionary containing the kirchhoff cost of every bus at
                all time slices. The keys are tuples of the label of the
                bus and the index of the time slice. The values are
                floats, representing the kirchhoff cost of the bus at
                the time slice, scaled by the KirchhoffFactor.
        """
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib, **self.calc_kirchhoff_cost_at_bus(
                bus=bus, result=solution, silent=silent)}
        return contrib

    def individual_kirchhoff_cost(self, solution, silent=True):
        """
        Returns a dictionary which contains the kirchhoff cost incurred
        at every bus at every time slice, without being scaled by the
        Kirchhofffactor
        
        Args:
            solution: (list)
                List of all qubits that have spin -1 in a solution.
            silent: (bool)
                Switch to enable status messages send to stdout. If
                true, no messages are sent.
                Default: True
        Returns:
            (dict)
                Dictionary containing the kirchhoff cost of every bus at
                all time slices. The keys are tuples of the label of the
                bus and the index of the time slice. The values are
                floats, representing the kirchhoff cost of the bus at
                the time slice, without being scaled by the
                Kirchhofffactor.
        """
        return {
            key: imbalance ** 2
            for key, imbalance in
            self.individual_power_imbalance(
                solution=solution, silent=silent).items()
        }

    def individual_power_imbalance(self,
                                   solution: list,
                                   silent: bool = True
                                   ) -> dict:
        """
        Returns a dictionary which contains the power imbalance at each
        bus at every time slice with respect to their type (too much or
        to little power) via its sign.
        
        Args:
            solution: (list)
                List of all qubits which have spin -1 in the solution.
            silent: (bool)
                Switch to enable status messages send to stdout. If
                true, no messages are sent.
                Default: True
        Returns:
            (dict)
                Dictionary containing the power imbalance of every bus
                at all time slices. The keys are tuples of the label of
                the bus and the index of the time slice. The values are
                floats, representing the power imbalance of the bus at
                the time slice, without being scaled by the
                Kirchhofffactor.
        """
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib, **self.calc_power_imbalance_at_bus(
                bus=bus, result=solution, silent=silent)}
        return contrib


class GlobalCostSquareWithSlack(GlobalCostSquare):
    """
    A subproblem class that models the minimization of the marginal
    costs. It does this by estimating it and then modelling the squared
    distance of the actual cost to the estimated cost. It also adds a
    slack term to the estimation which is independent of the network and
    serves to slightly adjust the estimation during the optimization.
    """
    # a dict to map config strings to functions which are used creating lists
    # of numbers, which can be used for weights of slack variables
    slack_representation_dict = {
        "binary_power": binary_power,
    }

    def __init__(self, backbone: IsingBackbone, config: dict):
        """
        A constructor for encoding marginal cost as quadratic penalties,
        including as well a slack variable.
        It inherits its functionality from the GlobalCostSquare
        constructor. Additionally, it adds slack qubits to the
        IsingBackbone.

        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance.
        """
        super().__init__(backbone=backbone, config=config)
        slack_weight_generator = self.slack_representation_dict[
            config.get("slack_type", "binary_power")]
        # an additional factor for scaling the weights of the qubits acting as
        # slack variables
        slack_scale = config.get("slack_scale", 1.0)
        # number of slack qubits used
        slack_size = config.get("slack_size", 7)
        slack_weights = [- slack_scale * i for i in
                         slack_weight_generator(slack_size)]
        # adding slack qubits with the label `slack_marginal_cost`
        self.backbone.create_qubit_entries_for_component(
            component_name="slack_marginal_cost",
            weights=slack_weights * len(backbone.snapshots),
            encoding_length=len(slack_weights)
        )

    # TODO refactor using a ising_backbone function for encoding squared distances
    def encode_marginal_costs(self, time: int) -> None:
        """
        The marginal costs of using generators are considered one single
        global constraint. The square of marginal costs is encoded
        into the energy and thus minimized.

        Args:
            time: (int)
                Index of time slice for which to estimate the marginal
                cost.

        Returns:
            (None)
                Modifies self.ising_coefficients. Adds to previously written
                interaction coefficient.
        """
        estimated_cost, offset = self.estimate_global_marginal_cost(
            time=time, expected_additonal_cost=0)
        self.print_estimation_report(estimated_cost=estimated_cost,
                                     offset=offset,
                                     time=time)
        generators = self.network.generators.index
        generators = list(generators) + ["slack_marginal_cost"]
        load = 0.0
        for bus in self.network.buses.index:
            load += self.backbone.get_load(bus, time)
        for first_generator in generators:
            if first_generator == "slack_marginal_cost":
                marginal_cost_first_generator = 1.
            else:
                marginal_cost_first_generator = \
                    self.network.generators["marginal_cost"].loc[first_generator] - offset
            for second_generator in generators:
                if second_generator == "slack_marginal_cost":
                    marginal_cost_second_generator = 1.
                else:
                    marginal_cost_second_generator = \
                        self.network.generators["marginal_cost"].loc[second_generator] - offset
                current_factor = self.scale_factor * \
                                 marginal_cost_first_generator * \
                                 marginal_cost_second_generator
                self.backbone.couple_components(
                    first_component=first_generator,
                    second_component=second_generator,
                    coupling_strength=current_factor
                )


class StartupShutdown(AbstractIsingSubproblem, ABC):
    pass
#    def encodeStartupShutdownCost(self, bus, time=0):
#        """
#        Adds the startup and shutdown costs for every generator attached to the bus. Those
#        costs are monetary costs incurred whenever a generator changes its status from one
#        time slice to the next. The first time slice doesn't incur costs because the status
#        of the generators before is unknown
#        
#        @param bus: str
#            label of the bus at which to add startup and shutdown cost
#        @param time: int
#            index of time slice which contains the generator status after a status change
#        @return: None
#            modifies self.problem. Adds to previously written interaction coefficient
#        """
#        # no previous information on first time step or when out of bounds

#        if time == 0 or time >= len(self.snapshots):
#            return
#
#        generators = self.get_bus_components(bus)['generators']
#
#        for generator in generators:
#            startup_cost = self.network.generators["start_up_cost"].loc[generator]
#            shutdown_cost = self.network.generators["shut_down_cost"].loc[generator]
#
#            # start up costs
#            # summands of (1-g_{time-1})  * g_{time})
#            self.couple_component_with_constant(
#                    generator,
#                    coupling_strength=self.monetaryCostFactor * startup_cost,
#                    time=time
#            )
#            self.couple_components(
#                    generator,
#                    generator,
#                    coupling_strength= -self.monetaryCostFactor * startup_cost,
#                    time = time,
#                    additional_time = time -1
#            )
#
#            # shutdown costs
#            # summands of g_{time-1} * (1-g_{time})
#            self.couple_component_with_constant(
#                    generator,
#                    coupling_strength=self.monetaryCostFactor * shutdown_cost,
#                    time=time-1
#            )
#            self.couple_components(
#                    generator,
#                    generator,
#                    coupling_strength= -self.monetaryCostFactor * shutdown_cost,
#                    time = time,
#                    additional_time = time -1
#            )
