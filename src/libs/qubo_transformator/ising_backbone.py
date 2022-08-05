"""This module is the central piece for the mathematical
model of the unit commitment problem as a QUBO/ising problem
It provides the data strucutre that contains the transformation
of network components into qubits and interactions of qubits

Expanding on the type of networks that can be modelled is achieved by
extending the `IsingBackbone`. 
"""

from typing import Any

import numpy as np

import pypsa
from numpy import ndarray


def binary_power_and_rest(number: int):
    """
    Constructs a minimal list of positive integers which sum up to the passed
    argument and such that for every smaller, positive number there exists
    a subtotal of the list that is equal to it.

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
    constraint.

    Modeling of various constraints is delegated to the instances of classes
    implementing the `IsingSubproblem` interface. An `IsingSubproblem`
    provides a method which adds the Ising representation of the constraint
    it models to the stored Ising problem in this class.

    Extending the Ising model of the network can be done in two ways. If
    you want to extend which kinds of networks can be read, you have to
    extend this class with methods that convert values of the network
    into qubits and write appropriate access methods. If you want to
    add a new constraint, you have to write a class that adheres to the
    `AbstractIsingSubproblem` interface
    """

    def __init__(self,
                 network: pypsa.Network,
                 configuration: dict):
        """
        Constructor for an Ising Backbone. It requires a network and
        the name of the function that defines how to encode lines. Then
        it goes through the configuration dictionary and encodes all
        sub-problem present into the instance.

        Args:
            network: (pypsa.Network)
                The pypsa network which to encode into qubits.
            configuration: (dict)
                A dictionary containing all subproblems to be encoded
                into an ising problem.
        """
        # network to be solved
        self.network = network
        if "snapshots" in configuration:
            self.network.snapshots = self.network.snapshots[:configuration.pop("snapshots")]
        self.snapshots = network.snapshots

        # contains ising coefficients
        self.ising_coefficients = {}
        self.ising_coefficients_positive = {}
        self.ising_coefficients_negative = {}
        # mirrors encodings of `self.ising_coefficients`, but is reset after encoding a
        # subproblem to get ising formulations of subproblems
        self.ising_coefficients_cached = {}

        # initializing data structures that encode the network into qubits
        # the encoding dict contains the mapping of network components to qubits
        self._qubit_encoding = {}
        # the weights dictionary contains a mapping of qubits to their weights
        self._qubit_weights = {}
        self.allocated_qubits = 0

        # read configuration dict, store in _subproblems and apply encodings
        self._subproblems = {}
        # dictionary of all support subproblems

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

    def flush_cached_problem(self) -> None:
        """
        Resets the cached changes of interactions.

        Returns:
            (None)
        """
        self.ising_coefficients_cached = {}

    # functions to couple components. The couplings are interpreted as
    # multiplications of QUBO polynomials. The final interactions are
    # coefficients for an ising spin glass problem
    def add_interaction(self, *args, weighted_interaction=True) -> None:
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
            args[:-1]: (list)
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
        if weighted_interaction:
            for qubit in key:
                interaction_strength *= self._qubit_weights[qubit]

        # if we couple two spins, we check if they are different. If both spins
        # are the same, we substitute the product of spins with 1, since
        # 1 * 1 = -1 * -1 = 1 holds. This makes it into a constant
        # contribution. Doesn't work for higher order interactions
        if len(key) == 2:
            if key[0] == key[1]:
                key = tuple([])
        # store interaction, wether it is positive or negative, and in the cache
        self.ising_coefficients[key] = self.ising_coefficients.get(key, 0) - interaction_strength
        if interaction_strength > 0:
            self.ising_coefficients_negative[key] = self.ising_coefficients_negative.get(key, 0) - interaction_strength
        else:
            self.ising_coefficients_positive[key] = self.ising_coefficients_positive.get(key, 0) - interaction_strength
        self.ising_coefficients_cached[key] = self.ising_coefficients_cached.get(key, 0) - interaction_strength

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
            time: (any)
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
            time: (any)
                Index of time slice of the first component for which to
                couple qubits representing it.
            additional_time: (any)
                Index of time slice of the second component for which
                to couple qubits representing it. The default parameter
                'None' is used if the time slices of both components
                are the same.
        Returns:
            (None)
                Modifies `self.ising_coefficients`. Adds to previously written
                interaction coefficient.

        Example:
            Let X_1, X_2 be the qubits representing first_component and
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

    def add_basis_polynomial_interaction(self,
                                         first_qubit: int = None,
                                         second_qubit: int = None,
                                         zero_qubits_list: list = None,
                                         interaction_strength: float = 1.0):
        """
        For a given list of qubo variables, adds the term to the ising interactions
        that has impact on the cost except for the state spefied in the `variable_value_list`

        This uses the fact, that a QUBO is a polynomial of order two. The space of valid
        QUBO's is then generated by the polynomials of the form
            (X_i - x_i)(X_j - x_j) for x_i, x_j in {0,1} and X_i, X_j arbitrary QUBO variables.

        Args:
            first_qubit: (int)
                The first qubit variable that is involved in the polynomial
            second_qubit: (int)
                The second qubit variable that is involved in the polynomial
            zero_qubits_list: (list)
                A list of qubits that are 0 (1 in the ising formulation) in the specified state
                of the QUBO variables to which the cost is added to.
            interaction_strength: (float)
                The magnitude of the term that is added to the QUBO

        Returns:
            (None)
                Modifies the stored ising problem
        """
        if zero_qubits_list is None:
            zero_qubits_list = []

        if first_qubit in zero_qubits_list:
            first_sign = -1
        else:
            first_sign = 1
        if second_qubit in zero_qubits_list:
            second_sign = -1
        else:
            second_sign = 1

        self.add_interaction(first_qubit, second_qubit, -0.25 * interaction_strength, weighted_interaction=False)
        self.add_interaction(first_qubit, second_sign * -0.25 * interaction_strength, weighted_interaction=False)
        self.add_interaction(second_qubit, first_sign * -0.25 * interaction_strength, weighted_interaction=False)
        self.add_interaction(first_sign * second_sign * -0.25 * interaction_strength, weighted_interaction=False)

    def encode_squared_distance(self,
                                label_dictionary: dict,
                                target: float = 0.0,
                                global_factor: float = 1.0,
                                time: any = None,
                                ) -> None:
        """
        Encodes the squared distance of all components in `label_dictionary.keys()`
        to the value `target` with respect to the factors given in keys.

        Args:
            label_dictionary: (dict)
                key, value pairs of all components involved in representing the
                target value.
            target: (float)
                the value that is the target for the sum of the components
                in the label_dictionary
            global_factor: (float)
                a factor by which all interaction are multiplied
            time: (any)
                the time step at which to couple the components

        Returns:
            (None)
                Modifies `self.ising_coefficients` and `self.cached_problem`
        
        """
        # constant contribution to cost function so that a configuration
        # that matches the target value has energy of 0
        self.add_interaction(global_factor * target ** 2)

        for first_component, first_factor in label_dictionary.items():
            factor = global_factor * first_factor

            self.couple_component_with_constant(first_component,
                                                - 2.0 * factor * target,
                                                time=time)
            for second_component, second_factor in label_dictionary.items():
                current_factor = factor * second_factor
                # attraction/repulsion term for different/same sign of power
                # at components
                self.couple_components(first_component,
                                       second_component,
                                       coupling_strength=current_factor,
                                       time=time)

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
        Returns how many non-zero interactions the ising problem has

        Returns:
            (int)
                number of ising interactions
        """
        return len(self.ising_coefficients)

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
        allocated to represent a network component, it also stored its weight
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
        snapshot_to_weight_dict = {
            snapshot: snapshot_to_weight_dict.get(snapshot, [])
            for snapshot in self.network.snapshots
        }

        # store component - snapshot - representing qubit relation
        self._qubit_encoding[component_name] = {
            snapshot: self.allocate_qubits_to_weight_list(snapshot_to_weight_dict[snapshot])
            for snapshot in self.network.snapshots
        }
        # expose qubit weights in `self._qubit_weights`
        for snapshot, qubit_list in self._qubit_encoding[component_name].items():
            for idx, qubit in enumerate(qubit_list):
                self._qubit_weights[qubit] = snapshot_to_weight_dict[snapshot][idx]

    def allocate_qubits_to_weight_list(self, weight_list: list):
        """
        For a given list of weights, returns a list of qubits which will we mapped
        to these weights, starting at the qubit `self.allocated_qubits` and increasing
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
                    self.get_generator_status(generator=generator,
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
        Returns the nominal power of a generator at a time step 
        
        Args:
            generator: (str)
                The generator label.
            time: (any)
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
        return max(self.network.generators.p_nom[generator] * p_max_pu, 0.0)

    def get_minimal_power(self, generator: str, time: any):
        """
        Returns the minimal power output of a generator at a time step
        """
        try:
            p_min_pu = self.network.generators_t.p_min_pu[generator].loc[time]
        except KeyError:
            p_min_pu = 0.0
        return max(self.network.generators.p_nom[generator] * p_min_pu, 0.0)

    def get_generator_status(self, generator: str, solution: list, time: any) -> float:
        """
        Returns the status of a generator which is the percentage of the
        maximum output at the time slice 'time' in a given solution.

        Args:
            gen: (str)
                The label of the generator.
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            time: (any)
                Index of time slice for which to get the generator
                status. This has to be in the network.snapshots index
        """
        maximum_output = self.get_nominal_power(generator, time)
        if maximum_output == 0.0:
            return 0.0
        generated_power = self.get_encoded_value_of_component(component=generator,
                                                              solution=solution,
                                                              time=time)
        return generated_power / maximum_output

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
                result[key] = self.get_generator_status(
                        generator=generator,
                        solution=solution,
                        time=time)
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
            time: (any)
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
            time: (any)
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
            time: (any)
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
            time: (any)
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

    def get_encoded_value_of_component(self,
                                       component: str,
                                       solution: list,
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
                marginal_cost += power_output_at_current_time \
                    * self.network.generators["marginal_cost"].loc[generator]
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
        for marginal_cost_at_bus in self.individual_marginal_cost(solution=solution).values():
            marginal_cost += marginal_cost_at_bus
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

    def get_hamiltonian_eigenvalues(self) -> tuple[ndarray, Any]:
        """
        Returns the eigenvalues and normalized eigenvectors of the
        hamiltonian matrix.

        Returns:
            (np.ndarray)
                A numpy array containing all eigenvalues.
        """
        return np.linalg.eigh(self.get_hamiltonian_matrix())
