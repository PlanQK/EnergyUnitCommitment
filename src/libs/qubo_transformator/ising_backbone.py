"""This module is the central piece for the mathematical
model of the unit commitment problem as a QUBO/ising problem
It provides the data structure that contains the transformation
of network components into qubits and interactions of qubits

Expanding on the type of networks that can be modelled is achieved by
extending the `IsingBackbone`.
"""

from typing import Any, Union

import numpy as np

import pypsa
from numpy import ndarray


class IsingBackbone:
    """
    This class implements the data structure for converting an optimization
    problem instance to an Ising spin glass problem.

    It acts as an endpoint to decode qubit configuration and encode
    coupling of QUBO variables into Ising interactions. For a specific
    problem instances, custom methods have to be written in a child class
    to access instance dependant information

    This class only acts as a problem agnostic data structure
    which other objects can use to model a specific constraint.
    It uses labels for groups of qubits that model one component of the problem.
    Those component can be indexed by a time slice, to model states of the
    same component at different points in time. Each qubit is designated a weight.
    The value of some encoded component is interpreted as the weighted sum of it's
    qubits.

    Modeling of various constraints is delegated to the instances of classes
    implementing the `IsingSubproblem` interface. An `IsingSubproblem`
    provides a method which adds the Ising representation of the constraint
    it models to the stored Ising problem in this class using the visitor pattern.

    Extending the Ising model can be done in two ways. If you want to implement
    problem specific methods, you can extend this class and add methods that
    additional access methods for interpreting states of qubits.

    You can add a new constraint by writing a class that adheres to the
    `AbstractIsingSubproblem` interface and have the instance of the qubo_transformator
    instantiate it in order to visit the data structure for encoding of the
    subproblem

    Attributes
        ----------
        _snapshots : list
            Contains a list of keys that are used for indexing the component qubits
            with a time component

        # contains ising coefficients
        _ising_coefficients : dict
            Contains the ising coefficients of the ising problem. The keys are ordered
            tuples containing the qubits of the respective interaction and strength as
            it's value

        _ising_coefficients_cached : dict
            This mirrors all encodings into `self._ising_coefficients` but is considering
            a working copy. This means that the complete ising problem will not use this.
            It is meant to be reset after an IsingSubproblem visits an instance of this class
        _qubit_encoding : dict
            This contains the mapping of components via their labels to qubits. Each key
            is the label of the component and the value is a dictionary. That dictionary
            has the values of the _snapshots attribute as the keys, and the list of qubits
            that represent the component at that snapshot.
        _qubit_weights  : dict
            A dictionary with integers as keys, which represent qubits, and their respective
            weight as the value. The weight of a qubit is a factor that is multiplied
            to any interaction this qubit is involved in unless it is explicitly turned off.
        _allocated_qubits : int
            The number of qubits that has been allocated to represent problem components
        _subproblems  : list
            A list of all `IsingSubproblem` instances that have visited this instance for
            encoding their subproblem into the ising model
    """

    def __init__(self):
        """
        Constructor for an Ising Backbone. It initializes various data structures
        that are used for modelling a problem but doesn't fill any of them. This has
        to be done by an `qubit_encoder.Encoder`, which can consume a data structure
        specifying an optimization problem and convert its components into qubits
        using string labels.

        Visiting `IsingSubproblem` instance encode the various constraints and the objective
        function of the optimization problem
        """
        # list of time slices that are modelled in the qubo. This assumes that the
        # problem components are time-independent.
        self._snapshots = [0]

        # contains ising coefficients
        self._ising_coefficients = {}
        # contains qubo coefficients
        self._qubo_coefficients = {}
        # mirrors all encodings into `self._ising_coefficients`. Resetting this to an empty
        # dictionary allows visiting subproblems to obtain an ising model of the subproblem
        # that it encodes
        self._ising_coefficients_cached = {}

        # initializing data structures that encode the problem components into qubits
        # the encoding dict contains the mapping of components via labels to qubits
        self._qubit_encoding = {}
        # this dictionary contains a mapping of qubits to their weights
        self._qubit_weights = {}
        self._allocated_qubits = 0

        # subproblems of the passed configuration are stored here
        self._subproblems = {}

    def __getattr__(self, method_name: str) -> callable:
        """
        This function delegates method calls to an IsingBackbone to a
        subproblem instance if IsingBackbone doesn't have such a method.

        We can use this by calling methods of IsingSubproblem instances
        that encoded their subproblem into this data structure. This delegates
        subproblem specific interpretation of qubit states to the IsingSubproblem
        If the name of the method is not unique among all
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
                    raise AttributeError(
                        f"{method_name} didn't resolve to " f"unique subproblem"
                    )
        if method:
            return method
        else:
            raise AttributeError(
                f"{method_name} was not found in any stored " f"subproblem"
            )

    def flush_cached_problem(self) -> None:
        """
        Resets the cached changes of interactions.

        Returns:
            (None)
        """
        self._ising_coefficients_cached = {}

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
        spin problem. Interactions are stored using an ordered tuple
        of the involved qubits and a float as the value

        The method can take an arbitrary number of arguments. The
        last argument is the interaction strength. All previous
        arguments are assumed to contain spin ids.

        Args:
            args[-1]: (float)
                The basic interaction strength before applying qubit
                weights.
            args[:-1]: (list)
                All qubits that are involved in this interaction.
            weighted_interaction: (bool)
                A flag for turning off application of qubit weights to
                the interaction

        Returns:
            (None)
                Modifies self._ising_coefficients by adding the strength of the
                interaction to it
        """
        if len(args) > 3:
            raise ValueError("Too many arguments for an interaction")
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
        # store interaction, whether it is positive or negative, and in the cache
        self._ising_coefficients[key] = (
            self._ising_coefficients.get(key, 0) - interaction_strength
        )
        self._ising_coefficients_cached[key] = (
            self._ising_coefficients_cached.get(key, 0) - interaction_strength
        )

    def couple_component_with_constant(
        self, component: str, coupling_strength: float = 1.0, time: any = None
    ) -> None:
        """
        Performs a QUBO multiplication involving a single variable on
        all qubits which are logically grouped to represent a component
        at a given time slice. This QUBO multiplication is translated
        into Ising interactions and then added to the currently stored
        Ising problem.

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
                Modifies self._ising_coefficients. Adds to previously written
                interaction coefficient.
        """
        if time is None:
            time = self._snapshots[0]
        component_address = self.get_representing_qubits(component, time)
        for qubit in component_address:
            # term with single spin after applying QUBO to Ising transformation
            self.add_interaction(qubit, 0.5 * coupling_strength)
            # term with constant cost contribution after applying QUBO to
            # Ising transformation
            self.add_interaction(0.5 * coupling_strength * self._qubit_weights[qubit])

    def couple_components(
        self,
        first_component: str,
        second_component: str,
        coupling_strength: float = 1,
        time: any = None,
        additional_time: any = None,
    ) -> None:
        """
        This method couples two labeled groups of qubits as a product
        according to their weight and the selected time step.
        It performs a QUBO multiplication involving exactly two
        components on all qubits which are logically grouped to
        represent these components at a given time slice. This QUBO
        multiplication is translated into Ising interactions, scaled by
        the coupling_strength and the respective weights of the qubits
        and then added to the currently stored Ising problem.

        Args:
            first_component: (str)
                The label of the first problem component.
            second_component: (str)
                The label of the second problem component.
            coupling_strength: (float)
                The scalar of QUBO multiplication by which to multiply all
                interactions.
            time: (any)
                The index of time slice of the first component for which to
                couple qubits representing it. It defaults to the first time step
                if no value is passed
            additional_time: (any)
                The index of time slice of the second component for which
                to couple qubits representing it. The default parameter
                'None' is used if the time slices of both components
                are the same.

        Returns:
            (None)
                Modifies `self._ising_coefficients`

        Example:
            Let 'x' and 'y' be the labels of two problem components. The
            component 'x' is represented by two qubits: x_1 and x_2. The
            component 'y' is represented by the qubits: y_1 and y_2. The
            method call
                `self.couple_components('x', 'y', 2.0)`
            will add the QUBO given by
            2 * (x_1 + x_2) * (y_1 + y_2) =   2 * x_1 * y_1
                                            + 2 * x_1 * y_2
                                            + 2 * x_2 * y_1
                                            + 2 * x_2 * y_2
            after translating it to interactions of the ising model using
            the translation rule x = (sigma-1)/2.
        """
        # Replace None default values with the first snapshot and
        # then figure out which qubits we want to couple based on the
        # component name and chosen time step
        if time is None:
            time = self._snapshots[0]
        if additional_time is None:
            additional_time = time
        first_component_address = self.get_representing_qubits(first_component, time)
        second_component_address = self.get_representing_qubits(
            second_component, additional_time
        )
        # components with 0 weight (power, capacity) vanish in the QUBO
        # formulation
        if (not first_component_address) or (not second_component_address):
            return
        # retrieving corresponding qubits is done. Now perform qubo
        # multiplication by expanding the product and add each summand
        # individually.
        for first_qubit in first_component_address:
            for second_qubit in second_component_address:
                # The body of this loop corresponds to the multiplication of
                # two QUBO variables. According to the QUBO - Ising
                # translation rule x = (sigma-1)/2 one QUBO multiplication
                # results in 4 ising interactions, including constants

                # term with two spins after applying QUBO to Ising
                # transformation if both spin ids are the same, this will
                # add a constant cost.
                # add_interaction performs substitution of spin with a constant
                self.add_interaction(
                    first_qubit, second_qubit, coupling_strength * 0.25
                )
                # terms with single spins after applying QUBO to Ising
                # transformation
                self.add_interaction(
                    first_qubit,
                    coupling_strength * self._qubit_weights[second_qubit] * 0.25,
                )
                self.add_interaction(
                    second_qubit,
                    coupling_strength * self._qubit_weights[first_qubit] * 0.25,
                )
                # term with constant cost contribution after applying QUBO to
                # Ising transformation
                self.add_interaction(
                    self._qubit_weights[first_qubit]
                    * self._qubit_weights[second_qubit]
                    * coupling_strength
                    * 0.25
                )

    def add_basis_polynomial_interaction(
        self,
        first_qubit: int = None,
        second_qubit: int = None,
        solution_list: list = None,
        interaction_strength: float = 1.0,
    ) -> None:
        """
        For two specified qubits, add an interaction that incurs no cost except if the two
        qubits have a particular state.

        This interaction uses that a QUBO is a polynomial of order two. The space of valid
        QUBOs is thus generated by the polynomials of the form
            (X_i - x_i)(X_j - x_j) for x_i, x_j in {0,1} and X_i, X_j arbitrary QUBO variables.
        Adding a polynomial of that form to the QUBO adds no cost if any of the qubits is a root
        of that polynomial. Only if this is not the case is a non-zero cost incurred

        Args:
            first_qubit: (int)
                The first qubit variable that is involved in the polynomial
            second_qubit: (int)
                The second qubit variable that is involved in the polynomial
            solution_list: (list)
                A list of qubits that are 1 in the specified state
                of the QUBO variables to which the cost is added to. For our Ising formulation
                they are the spin id's with spin -1
            interaction_strength: (float)
                The magnitude of the term that is added to the QUBO

        Returns:
            (None)
                Modifies the stored ising problem

        Example:
            For a backbone instance with an empty `_ising_coefficients` attribute, the
            method call
                self.add_basis_polynomial_interaction(first_qubit=0,
                                                      second_qubit=1,
                                                      solution_list=[1],
                                                      interaction_strength=1.0)
            adds an interaction such that calling self.calc_cost will return the following cost:
                self.calc_cost([]) == 0
                self.calc_cost([0]) == 0
                self.calc_cost([1]) == 1
                self.calc_cost([0,1]) == 0
        """
        if solution_list is None:
            solution_list = []
        interaction_strength = (-1) ** len(solution_list) * 0.25 * interaction_strength
        first_sign, second_sign = -1, -1
        if first_qubit in solution_list:
            first_sign *= -1
        if second_qubit in solution_list:
            second_sign *= -1
        self.add_interaction(
            first_qubit, second_qubit, interaction_strength, weighted_interaction=False
        )
        self.add_interaction(
            first_qubit, second_sign * interaction_strength, weighted_interaction=False
        )
        self.add_interaction(
            second_qubit, first_sign * interaction_strength, weighted_interaction=False
        )
        self.add_interaction(
            first_sign * second_sign * interaction_strength, weighted_interaction=False
        )

    def encode_squared_distance(
        self,
        label_dictionary: dict = None,
        label_list: list = None,
        target: float = 0.0,
        global_factor: float = 1.0,
        time: any = None,
    ) -> None:
        """
        Encodes the squared distance of all components in the `label_dictionary` and
        label_list to the value `target` with respect to the factors given in keys.

        Args:
            label_dictionary: (dict)
                a dictionary with key, value pairs of all components involved in representing the
                target value. The value is a float and represent the qubit's weight
            label_list: (list)
                A list of labels that are added as an interaction with weight 1.0
            target: (float)
                the value that is the target for the sum of the components
                in the label_dictionary and label_list
            global_factor: (float)
                a factor by which all interaction are multiplied
            time: (any)
                the time step at which to couple the components

        Returns:
            (None)
                Modifies `self._ising_coefficients` and `self.cached_problem`
        """
        if label_dictionary is None:
            label_dictionary = {}
        if label_list is None:
            label_list = []
        label_dictionary = {**label_dictionary, **{label: 1.0 for label in label_list}}
        # constant contribution to cost function so that a configuration
        # that matches the target value has energy of 0
        self.add_interaction(global_factor * target**2)

        for first_component, first_factor in label_dictionary.items():
            factor = global_factor * first_factor

            self.couple_component_with_constant(
                component=first_component,
                coupling_strength=2.0 * factor * target,
                time=time,
            )
            for second_component, second_factor in label_dictionary.items():
                current_factor = factor * second_factor
                # attraction/repulsion term for different/same sign of power
                # at components
                self.couple_components(
                    first_component,
                    second_component,
                    coupling_strength=current_factor,
                    time=time,
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
        return self._allocated_qubits

    def num_interactions(self) -> int:
        """
        Returns how many non-zero interactions the ising problem has

        Returns:
            (int)
                number of ising interactions
        """
        return len(self._ising_coefficients)

    def create_qubit_entries_for_component(
        self,
        component_name: str,
        snapshot_to_weight: Union[dict, list] = None,
    ) -> None:
        """
        A function to create qubits in the data structure that can be accessed
        as a group via its label.

        The method places several restriction on what it accepts in
        order to generate a valid QUBO later on. The checks are intended
        to prevent name or qubit collision.

        Allocating qubits for a component modifies the attributes
            `_qubit_encoding, _qubit_weight, _allocated_qubits`
        and requires to specify how many qubits with which weight represents
        the component. It raises an error if the label has already being used
        for a group of qubits.

        The number of qubits and weights can be specified as:
            - a dictionary of lists with snapshots as keys and the corresponding value being the
              weights of the qubits representing the component at that snapshot.
            - a list of weights. It is assumed that the number of representing qubits
              and their weights are constant across all snapshots
            - the default value `None`. This means that the component is represented
              by a single qubit with weight `1.0` at each snapshot

        Args:
            component_name: (str)
                The string used to couple the component with qubits.
            snapshot_to_weight: (dict|list)
                A dictionary or list to specify qubit weights at each snapshot

        Returns:
            (None)
                Modifies `self._qubit_weights, self._qubit_encoding, self._allocated_qubits`
        """
        if component_name in self._qubit_encoding:
            raise ValueError("Component name has already been used")

        if snapshot_to_weight is None:
            snapshot_to_weight_dict = {snapshot: 1.0 for snapshot in self._snapshots}
        elif isinstance(snapshot_to_weight, list):
            snapshot_to_weight_dict = {
                snapshot: snapshot_to_weight for snapshot in self._snapshots
            }
        else:
            snapshot_to_weight_dict = {
                snapshot: snapshot_to_weight.get(snapshot, [])
                for snapshot in self._snapshots
            }

        # store component - snapshot - representing qubit relation
        self._qubit_encoding[component_name] = {
            snapshot: self.allocate_qubits_to_weight_list(
                snapshot_to_weight_dict[snapshot]
            )
            for snapshot in self._snapshots
        }
        # expose qubit weights in `self._qubit_weights`
        for snapshot, qubit_list in self._qubit_encoding[component_name].items():
            for idx, qubit in enumerate(qubit_list):
                self._qubit_weights[qubit] = snapshot_to_weight_dict[snapshot][idx]

    def allocate_qubits_to_weight_list(self, weight_list: list) -> list:
        """
        For a given list of weights, returns a list of qubits which will we mapped
        to these weights, starting at the qubit `self._allocated_qubits` and increasing
        that counter appropriately

        Args:
            weight_list: (list[float])
                A list of floats, which describes weights to be mapped to newly allocated qubits

        Returns:
            (list[int])
                a list of consecutive integers, which represent qubits which are mapped
                to the weight list. Also increases the internal qubit count
        """
        num_new_allocated_qubits = len(weight_list)
        allocated_qubit_list = list(
            range(
                self._allocated_qubits,
                self._allocated_qubits + num_new_allocated_qubits,
            )
        )
        self._allocated_qubits += num_new_allocated_qubits
        return allocated_qubit_list

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

    def get_representing_qubits(self, component: str, time: any = None) -> list:
        """
        Returns a list of all qubits that are grouped under a label at
        a given time step.

        Args:
            component: (str)
                The label of the qubit group
            time: (any)
                The time step for which to return the qubits that are labeled
                by the passed `component` argument

        Returns:
            (list)
                A list of qubits that are grouped under the component label at the
                passed time step
        """
        if time is None:
            time = self._snapshots[0]
        return self._qubit_encoding[component][time]

    def get_qubit_mapping(self, time: any = None) -> dict:
        """
        Returns the dictionary with all labels and the qubits were used for
        representation in an ising problem for a particular time step

        Args:
            time: (any)
                Index of time slice for which to get qubit map

        Returns:
            (dict)
                A dictionary of all network components and their qubits. The
                labels are the keys and the list of qubits for that label
                at the specified time step are the values
        """
        return {
            component: self.get_representing_qubits(component=component, time=time)
            for component in self._qubit_encoding.keys()
        }

    def get_interaction(self, *qubits) -> float:
        """
        Returns the interaction coefficient of the passed qubits.

        Args:
            *args: (list[int])
                All qubits that are involved in this interaction.

        Returns:
            (float)
                The interaction strength between all qubits in args.
        """
        return self._ising_coefficients.get(tuple(sorted(set(qubits))), 0.0)

    def get_encoded_value_of_component(
        self, component: str, solution: list, time: any = 0
    ) -> float:
        """
        Returns the encoded value of a component according to the spin
        configuration in the solution at a given time slice.

        A component is represented by a list of weighted qubits. The
        encoded value is the weighted sum of all active qubits. The active
        qubits are exactly those qubits, that are in the `solution` argument

        Args:
            component: (str)
                A label of the component for which to retrieve the encoded value.
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            time: (any)
                The time step at which to retrieve the encoded value.

        Returns:
            (float)
                The value of the component encoded in the spin configuration of solution.
        """
        value = 0.0
        for qubit in self._qubit_encoding[component][time]:
            if qubit in solution:
                value += self._qubit_weights[qubit]
        return value

    def calc_cost(self, solution: list, ising_interactions: dict = None) -> float:
        """
        Calculates the energy of a spin state including the constant
        energy contribution.

        The default ising spin glass state that is used to calculate
        the energy of a solution is the full problem stored in the
        IsingBackbone. If an interaction map is passed via the
        `ising_interactions` argument, those are used instead to
        evaluate the cost of a solution.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            ising_interactions: (dict)
                The ising problem to be used to calculate the cost. The
                default value is to use the stored ising problem

        Returns:
            (float)
                The energy of the spin glass state in solution.
        """
        solution = set(solution)
        if ising_interactions is None:
            ising_interactions = self._ising_coefficients
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
        return [
            (v, list(k))
            for k, v in self._ising_coefficients.items()
            if v != 0 and len(k) > 0
        ]

    def get_hamiltonian_matrix(self) -> list:
        """
        Returns a matrix containing the ising hamiltonian

        Returns:
            (list)
                A list of lists representing the hamiltonian matrix.
        """
        qubits = range(self._allocated_qubits)
        hamiltonian = [[self.get_interaction(i, j) for i in qubits] for j in qubits]
        return hamiltonian

    def get_hamiltonian_eigenvalues(self) -> tuple[ndarray, Any]:
        """
        Returns the eigenvalues and normalized eigenvectors of the
        hamiltonian matrix calculating by using numpy.

        Returns:
            (np.ndarray)
                A numpy array containing all eigenvalues.
        """
        return np.linalg.eigh(self.get_hamiltonian_matrix())

    def get_spectral_gap(self):
        """
        Calculates the difference of the smallest eigenvalues of
        the problem hamiltonian

        Returns: (float)
            The spectral gap of the problem hamiltonian
        """
        sorted_eigenvalues = sorted(self.get_hamiltonian_eigenvalues()[0])
        return sorted_eigenvalues[1] - sorted_eigenvalues[0]

    def generate_report(self, solution: list) -> dict:
        """
        For the given solution, calculates various properties of the
        solution.

        Args:
            solution: (list)
                A list of all qubits that have spin -1 in a solution.
        Returns:
            (dict)
                A dictionary containing general information about the
                solution.
        """
        return {
            "total_cost": self.calc_cost(solution=solution),
            "spectral_gap": self.get_spectral_gap()
        }
    def get_snapshots(self) -> list:
        """
        Returns the list of time slices modeled in the qubo.

        Returns:
            (list)
                The list of indices indexing the time slices modeled.
        """
        return self._snapshots
    def get_ising_coefficients(self) -> dict:
        """
        Returns the dictionary storing the Ising coefficients.

        Returns:
            (dict)
                The dictionary with ordered tuples of qubits as keys and their
                Ising coefficients as values.
        """
        return self._ising_coefficients
    def get_ising_coefficients_cached(self) -> dict:
        """
        Returns the dictionary storing the cached version of the Ising coefficients.

        Returns:
            (dict)
                The dictionary with ordered tuples of qubits as keys and their
                cached Ising coefficients as values.
        """
        return self._ising_coefficients_cached
    def get_qubit_weights(self) -> dict:
        """
        Returns the dictionary storing the weight of each qubit.

        Returns:
            (dict)
                The dictionary with qubits as keys and their respective
                weight as values.
        """
        return self._qubit_weights
    def get_subproblems(self) -> list:
        """
        Returns the list storing all Ising subproblems that have visited this
        instance for encoding their subproblem.

        Returns:
            (list)
                The list of IsingSubproblem instances which have been encoded.
        """
        return self._subproblems

class NetworkIsingBackbone(IsingBackbone):
    """
    This class implements the conversion of a unit commitment problem
    given by a Pypsa network to an Ising spin glass problem.
    It extends the ising backbone to get pypsa data like
    loads, generators at buses based on their label and so on.
    """

    def __init__(self, network: pypsa.Network):
        """
        Constructor for an Ising Backbone. It requires a network and
        the name of the function that defines how to encode lines. Then
        it goes through the configuration dictionary and encodes all
        sub-problem present into the instance.

        Args:
            network: (pypsa.Network)
                The pypsa network which to encode into qubits.
        """
        super().__init__()
        # network to be solved
        self.network = network
        # list of time slices that are modelled in the qubo
        self._snapshots = self.network.snapshots

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
                A copy of the attribute `network` in which time-dependant
                values for the generators and lines are set according to
                the given solution.
        """
        output_network = self.network.copy()
        # get Generator/Line Status
        for time in self._snapshots:
            for generator in output_network.generators.index:
                # set value in status-dataframe in generators_t dictionary
                status = int(
                    self.get_generator_status(
                        generator=generator, solution=solution, time=time
                    )
                )
                column_status = list(output_network.generators_t.status.columns)
                if generator in column_status:
                    index_generator = column_status.index(generator)
                    output_network.generators_t.status.iloc[
                        time, index_generator
                    ] = status
                else:
                    output_network.generators_t.status[generator] = status

                # set value in p-dataframe in generators_t dictionary
                p = self.get_encoded_value_of_component(
                    component=generator, solution=solution, time=time
                )
                columns_p = list(output_network.generators_t.p.columns)
                if generator in columns_p:
                    index_generator = columns_p.index(generator)
                    output_network.generators_t.p.iloc[time, index_generator] = p
                else:
                    output_network.generators_t.p[generator] = p

                # set value in p_max_pu-dataframe in generators_t dictionary
                columns_p_max_pu = list(output_network.generators_t.p_max_pu.columns)
                p_nom = output_network.generators.loc[generator, "p_nom"]
                if p == 0:
                    p_max_pu = 0.0
                else:
                    p_max_pu = p_nom / p
                if generator in columns_p_max_pu:
                    index_generator = columns_p_max_pu.index(generator)
                    output_network.generators_t.p_max_pu.iloc[
                        time, index_generator
                    ] = p_max_pu
                else:
                    output_network.generators_t.p_max_pu[generator] = p_max_pu

            for line in output_network.lines.index:
                encoded_val = self.get_encoded_value_of_component(
                    component=line, solution=solution, time=time
                )
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
                    'generators': A list of labels of generators that
                                  are at the bus.
                    'positive_lines': A list of labels of lines that end
                                      in this bus.
                    'negative_lines': A list of labels of lines that start
                                      in this bus.
        """
        if bus not in self.network.buses.index:
            raise ValueError("the bus " + bus + " doesn't exist")
        return {
            "generators": list(
                self.network.generators[self.network.generators.bus == bus].index
            ),
            "positive_lines": list(
                self.network.lines[self.network.lines.bus1 == bus].index
            ),
            "negative_lines": list(
                self.network.lines[self.network.lines.bus0 == bus].index
            ),
        }

    def get_nominal_power(self, generator: str, time: any) -> float:
        """
        Returns the nominal power of a generator at a time step

        Args:
            generator: (str)
                The generator label
            time: (any)
                The snapshot at which to get the nominal power. It has to be
                in the network.snapshot index

        Returns:
            (float)
                Nominal power available at `generator` at time slice `time`
        """
        try:
            p_max_pu = self.network.generators_t.p_max_pu[generator].loc[time]
        except KeyError:
            p_max_pu = 1.0
        return max(self.network.generators.p_nom[generator] * p_max_pu, 0.0)

    def get_minimal_power(self, generator: str, time: any) -> float:
        """
        Returns the minimal power output of a generator at a time step

        Args:
            generator: (str)
                The generator label
            time: (any)
                The snapshot at which to get the minimal power output. It
                has to be in the network.snapshot index

        Returns:
            (float)
                Minimal power output of `generator` at time slice `time` if
                it is committed
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
            generator: (str)
                The label of the generator.
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            time: (any)
                Index of time slice for which to get the generator status.
                This has to be in the network.snapshots index

        Return:
            (float)
                The percentage of the nominal power output of the generator
                at the time step according to the given qubit solution
        """
        maximum_output = self.get_nominal_power(generator, time)
        if maximum_output == 0.0:
            return 0.0
        generated_power = self.get_encoded_value_of_component(
            component=generator, solution=solution, time=time
        )
        return generated_power / maximum_output

    def get_generator_dictionary(self, solution: list, stringify: bool = True) -> dict:
        """
        Builds a dictionary containing the status of all generators at
        all time slices for a given solution of qubit spins.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            stringify: (bool)
                If this is true, dictionary keys are cast as strings, so
                they can for json

        Returns:
            (dict)
                A dictionary containing the status of all generators at
                all time slices. The keys are either tuples of the
                label of the generator and the index of the time slice,
                or these tuples typecast as strings, depending on the
                'stringify' argument. The values are booleans, encoding
                the status of the generators at the time slice.
        """
        result = {}
        for generator in self.network.generators.index:
            for time in self._snapshots:
                key = (generator, time)
                if stringify:
                    key = str(key)
                result[key] = self.get_generator_status(
                    generator=generator, solution=solution, time=time
                )
        return result

    def get_flow_dictionary(self, solution: list, stringify: bool = True) -> dict:
        """
        Builds a dictionary containing all power flows at all time
        slices for a given solution of qubit spins.

        Args:
            solution: (list)
                A list of all qubits which have spin -1 in the solution.
            stringify: (bool)
                If this is true, dictionary keys are cast as strings, so
                they can for json

        Returns:
            (dict)
                A dictionary containing the flow of all lines at
                all time slices. The keys are either tuples of the
                label of the generator and the index of the time slice,
                or these tuples typecast as strings, depending on the
                'stringify' argument. The values are floats,
                representing the flow of the lines at the time slice.
        """
        result = {}
        for line_id in self.network.lines.index:
            for time in self._snapshots:
                key = (line_id, time)
                if stringify:
                    key = str(key)
                result[key] = self.get_encoded_value_of_component(
                    component=line_id, solution=solution, time=time
                )
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
                The total load at 'bus' at time step 'time'.
        """
        loads_at_current_bus = self.network.loads[self.network.loads.bus == bus].index
        all_loads = self.network.loads_t["p_set"].loc[time]
        result = all_loads[all_loads.index.isin(loads_at_current_bus)].sum()
        if result == 0:
            if not silent:
                print(
                    f"Warning: No load at {bus} at timestep {time}.\n"
                    f"Falling back to constant load"
                )
            all_loads = self.network.loads["p_set"]
            result = all_loads[all_loads.index.isin(loads_at_current_bus)].sum()
        if result < 0:
            raise ValueError("negative Load at current bus")
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
        return {**super().generate_report(solution=solution),
            # "total_cost": self.calc_cost(solution=solution),
            "kirchhoff_cost": self.calc_kirchhoff_cost(solution=solution),
            "kirchhoff_cost_by_time": self.calc_kirchhoff_cost_by_time(
                solution=solution
            ),
            "power_imbalance": self.calc_power_imbalance(solution=solution),
            "total_power": self.calc_total_power_generated(solution=solution),
            "marginal_cost": self.calc_marginal_cost(solution=solution),
            "individual_kirchhoff_cost": self.individual_cost_contribution(
                solution=solution
            ),
            "unit_commitment": self.get_generator_dictionary(
                solution=solution, stringify=True
            ),
            "powerflow": self.get_flow_dictionary(solution=solution, stringify=True),
        }

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
            contrib = {
                **contrib,
                **self.calc_marginal_cost_at_bus(bus=bus, solution=solution),
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
                a string, and the values are the marginal costs at this
                bus at this time slice.
        """
        contrib = {}
        for time in self._snapshots:
            marginal_cost = 0.0
            components = self.get_bus_components(bus)
            for generator in components["generators"]:
                power_output_at_current_time = self.get_encoded_value_of_component(
                    component=generator, solution=solution, time=time
                )
                marginal_cost += (
                    power_output_at_current_time
                    * self.network.generators["marginal_cost"].loc[generator]
                )
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
        for marginal_cost_at_bus in self.individual_marginal_cost(
            solution=solution
        ).values():
            marginal_cost += marginal_cost_at_bus
        return marginal_cost
