import typing

from abc import ABC

import pypsa

from .ising_backbone import IsingBackbone


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


class AbstractIsingSubproblem:
    """
    An interface for classes that model the Ising formulation
    subproblem of a unit commitment problem. Classes that model
    a subproblem/constraint are subclasses of this class and
    adhere to the following structure. Each subproblem or
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
        self.scale_factor = config.setdefault("scale_factor", 1.0)
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
    a unit commitment problem.
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

    def __init__(self, backbone: IsingBackbone, config: dict):
        """
        Since all marginal cost use an offset of the marginal cost to center
        it around zero, initializing the offset from the config is done here.

        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the marginal cost problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance of the marginal costs problem.
        """
        super().__init__(backbone, config)
        self.range_factor = self.get_range_factor(config)
        if "target_cost" in config:
            self.set_from_target_cost(config["target_cost"])
        else:
            self.set_from_offset(config["offset"])
        print(f"\n--- Encoding marginal costs: {config['strategy']} ---")
        print(f"Marginal cost offset: {self.offset}")
        print(f"Equivalent offset fixed cost: {self.target_cost}\n")

    def set_from_target_cost(self, target_cost: float):
        """
        Calculates the corresponding marginal cost offset to the given
        target and writes both as attributes to the object.

        It is chosen in a way that a solution satisfying the kirchhoff constraint
        with cost equal to the target with respect to the original cost function
        incurs 0 cost with respect to the transformed function

        Args:
            target_cost: (float)
                The marginal cost to which to minimize the squared distance
        """
        self.target_cost = target_cost
        # not applicable for multisnapshot networks
        self.offset = self.target_cost / float(self.backbone.get_total_load(self.network.snapshots[0]))

    def set_from_offset(self, offset: float):
        """
        Calculates the corresponding marginal cost target to a given
        offset and writes both as attributes to the object

        The target cost is calucalted such thata solution satisfying the kirchhoff 
        constraint with cost equal to the target cost with respect to the original 
        cost function incurs 0 cost with respect to the transformed function

        Args:
            offset: (float)
                A float by which to offset the marginal costs of generators
                per unit of power produced
        """
        self.offset = offset
        # not applicable for multisnapshot networks
        self.target_cost = offset * float(self.backbone.get_total_load(self.network.snapshots[0]))


class MarginalAsPenalty(MarginalCostSubproblem):
    """
    A subproblem class that models the minimization of the marginal
    costs. It does this by adding a penalty to each qubit of a generator
    with the value being the marginal costs incurred by committing that
    generator. This linear penalty can be slightly changed.
    """

    def encode_subproblem(self) -> None:
        """
        Encodes the minimization of the marginal cost by applying a
        penalty at each qubit of a generator equal to the cost it would
        incur if committed.

        Returns:
            (None)
                Modifies self.backbone.
        """
        # Marginal costs are only modelled as linear penalty. Thus, it suffices 
        # to iterate over all time steps and all buses to get all generators
        for time in self.network.snapshots:
            for bus in self.network.buses.index:
                self.encode_marginal_costs(bus=bus, time=time)

    def encode_marginal_costs(self, bus: str, time: any) -> None:
        """
        Encodes marginal costs for running generators and transmission
        lines at a single bus.
        This uses an offset calculated in ´marginal_cost_offset´, which is
        dependent on all generators of the entire network for a single
        time slice.

        Args:
            bus: (str)
                Label of the bus at which to add marginal costs.
            time: (any)
                Index of time slice for which to add marginal cost.

        Returns:
            (None)
                Modifies self.ising_coefficients. Adds to previously written
                interaction coefficient.
        """
        generators = self.backbone.get_bus_components(bus)['generators']
        marginal_cost_df = self.network.generators["marginal_cost"]
        for generator in generators:
            self.backbone.couple_component_with_constant(
                component=generator,
                coupling_strength=self.scale_factor * (marginal_cost_df.loc[generator] - self.offset),
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
        Reads the additional `line_cost_factor` variable which is uses to 
        set a marginal costs equivalency for transmitted power

        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the marginal cost problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance of the marginal costs problem.
        """
        super().__init__(backbone, config)
        self.line_cost_factor = config.setdefault("line_cost_factor", 1.0)

    def encode_subproblem(self) -> None:
        """
        Encodes the minimization of the marginal cost by adding an offset 
        to the marginal costs. Then at each bus the marginal costs are modeled
        as the squared distance of the offset cost to zero.

        Returns:
            (None)
                Modifies `self.backbone`.
        """
        # Estimation is done independently at each bus. Thus, it suffices to
        # iterate over all snapshots and buses to encode the subproblem
        for time in self.network.snapshots:
            for bus in self.network.buses.index:
                self.encode_marginal_costs(bus=bus, time=time)

    def get_generator_to_cost_dict(self, bus: str) -> dict:
        """
        Returns a dictionary with generators of the specified bus as keys and 
        the their offset marginal costs

        Args:
            bus: (str)
                The label of the bus of the network

        Returns:
            (dict)
                A dict with generators as keys and their offset marginal costs as 
                values
        """
        generators = self.network.generators
        return {
            generator: self.network.generators["marginal_cost"].loc[generator] - self.offset
            for generator in generators.index[generators["bus"] == bus]
        }

    def encode_marginal_costs(self, bus: str, time: any) -> None:
        """
        Encodes marginal costs at a bus by first estimating a lower
        bound of unavoidable marginal costs, then deviations in the
        marginal cost from that estimation are penalized quadratically.

        Args:
            bus: (str)
                Label of the bus at which to encode marginal costs.
            time: (any)
                Index of the time slice for which to estimate marginal
                costs.

        Returns:
            (None)
                Modifies self.backbone. Adds to previously written
                interaction coefficient
        """
        components = self.backbone.get_bus_components(bus)
        bus_components_to_cost = self.get_generator_to_cost_dict(bus)
        bus_components_to_cost.update({
            label: self.line_cost_factor
            for label in components['positive_lines']
        })
        bus_components_to_cost.update({
            label: - self.line_cost_factor
            for label in components['negative_lines']
        })
        self.backbone.encode_squared_distance(
            bus_components_to_cost,
            global_factor=self.scale_factor,
            time=time,
        )


class GlobalCostSquare(MarginalCostSubproblem):
    """
    A subproblem class that models the minimization of the marginal
    costs. It does this by estimating it and then modelling the squared
    distance of the actual cost to the estimated cost.
    """

    def __init__(self, backbone: IsingBackbone, config: dict):
        """
        Since all marginal cost use an offset of the marginal cost to center
        it around zero, initializing the offset from the config is done here.

        Args:
            backbone: (IsingBackbone)
                The backbone on which to encode the marginal cost problem.
            config: (dict)
                A dict containing all necessary configurations to
                construct an instance of the marginal costs problem.
        """
        super().__init__(backbone, config)

    def get_range_factor(self, config: dict):
        """
        Get linear factor of linear transformation of marginal costs
        defaults to a value of 1.0

        Args:
            config: (dict)
                the dict containing the configuration data
        """
        return config.setdefault("range_factor", 1.0)

    def calc_transformed_target_value(self, time):
        """
        Calculates the corresponding target value to the cost given in the
        original configuration with respect to the linear transformation of
        the marginal costs.

        For this class, the only valid target after the transformation is `0.0`.
        This corresponds to the cost given by the product of the total load
        and the offset
        """
        return 0.0

    def calc_offset_cost(self, generator):
        """
        For a given generator, returns the marginal cost of that generator after
        applying the linear transformation given in the config of this subproblem

        Args:
            generator: (str)
                The label of a self.network generator

        Returns:
            The transformed marginal cost of that generator
        """
        return (self.network.generators["marginal_cost"][generator] - self.offset)  \
                * self.range_factor 

    def calc_transformed_marginal_costs(self, time) -> dict:
        """
        Returns a dictionary with generators as keys and the their offset marginal
        costs
        """
        return {generator: self.calc_offset_cost(generator) for generator in self.network.generators.index}

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


    def encode_marginal_costs(self, time: any) -> None:
        """
        The marginal costs of using generators are considered one single
        global constraint. The square of marginal costs is encoded
        into the energy and thus minimized.

        Args:
            time: (any)
                Index of time slice for which to encode marginal costs.

        Returns:
            (None)
                Modifies self.ising_coefficients. Adds to previously written
                interaction coefficient.
        """
        self.backbone.encode_squared_distance(
            self.calc_transformed_marginal_costs(time),
            target=self.calc_transformed_target_value(time),
            global_factor=self.scale_factor,
            time=time,
        )


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
        "binary_power": binary_power_and_rest,
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
        self.slack_scale = config.get("slack_scale", 1.0)
        # number of slack qubits used
        self.slack_size = config.get("slack_size", 3)
        self.slack_weights = [-weight for weight in slack_weight_generator(self.slack_size)]

        snapshot_to_slack_dict = {
            snapshot: self.slack_weights
            for snapshot in self.network.snapshots
        }
        # adding slack qubits with the label `slack_marginal_cost`
        self.backbone.create_qubit_entries_for_component(
            component_name="slack_marginal_cost",
            snapshot_to_weight_dict=snapshot_to_slack_dict
        )

    def calc_transformed_marginal_costs(self, time) -> dict:
        """
        Returns a dictionary with generators as keys and the their offset marginal
        costs. Adds the slack variable as if it was a generator
        i
        Return:
            (dict)
                A dictionary with generators, the slack component and associated
                marginal costs as values
        """
        result = super().calc_transformed_marginal_costs(time=time)
        result["slack_marginal_cost"] = self.slack_scale
        return result


class KirchhoffSubproblem(AbstractIsingSubproblem):
    """
    A class that models the Kirchhoff subproblem of a unit commitment
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
        print(f"--- Encoding kirchhoff constraints")

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
        self.ising_coefficients = self.backbone.ising_coefficients_cached

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
            time: (any)
                Index of time slice at which to enforce the kirchhoff
                constraint.

        Returns:
            (None)
                Modifies self.ising_coefficients. Adds to previously written
                interaction coefficient.
        """
        components = ising_backbone.get_bus_components(bus)
        label_dictionary = {
            label: 1.0
            for label in components['generators'] + components['positive_lines']
        }
        label_dictionary.update({
            label: -1.0
            for label in components['negative_lines']
        })

        demand = ising_backbone.get_load(bus, time=time)

        ising_backbone.encode_squared_distance(
            label_dictionary=label_dictionary,
            target=-demand,
            global_factor=self.scale_factor,
            time=time
        )
        return

    def calc_power_imbalance_at_bus_at_time(self,
                                            bus: str,
                                            time: any,
                                            result: list,
                                            ) -> float:
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
            (float)
                The deviation from the kirchhoff constraint at the given bus and time
                step by the solution of the optimization given in the result list
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
            time: (any)
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
                                             result=result).items()
        }

    def calc_kirchhoff_cost_by_time(self, solution: list) -> dict:
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
            # cast to str if it is not admissable as a json key
            if isinstance(time, int):
                result[time] = kirchhoff_cost
            else:
                result[str(time)] = kirchhoff_cost
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
                                     ) -> dict:
        """
        Returns a dictionary which contains the kirchhoff costs incurred
        at all busses for every time slice, scaled by the
        KirchhoffFactor.

        Args:
            solution: (list)
                List of all qubits that have spin -1 in a solution.

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
                bus=bus, result=solution)}
        return contrib

    def individual_kirchhoff_cost(self, solution):
        """
        Returns a dictionary which contains the kirchhoff cost incurred
        at every bus at every time slice, without being scaled by the
        lagrange multiplier for the kirchhoff constraint.
        
        Args:
            solution: (list)
                List of all qubits that have spin -1 in a solution.

        Returns:
            (dict)
                Dictionary containing the kirchhoff cost of every bus at
                all time slices. The keys are tuples of the label of the
                bus and the index of the time slice. The values are
                floats, representing the kirchhoff cost of the bus at
                the time slice, without being scaled by the lagrange multiplier
                for the kirchhoff constraint.
        """
        return {
            key: imbalance ** 2
            for key, imbalance in
            self.individual_power_imbalance(
                solution=solution).items()
        }

    def individual_power_imbalance(self,
                                   solution: list,
                                   ) -> dict:
        """
        Returns a dictionary which contains the power imbalance at each
        bus at every time slice with respect to their type (too much or
        to little power) via its sign.
        
        Args:
            solution: (list)
                List of all qubits which have spin -1 in the solution.

        Returns:
            (dict)
                Dictionary containing the power imbalance of every bus
                at all time slices. The keys are tuples of the label of
                the bus and the index of the time slice. The values are
                floats, representing the power imbalance of the bus at
                the time slice, without being scaled by the lagrange
                multiplier for the kirchhoff constraint.
        """
        contrib = {}
        for bus in self.network.buses.index:
            contrib = {**contrib, **self.calc_power_imbalance_at_bus(
                bus=bus, result=solution)}
        return contrib


class MinimalGeneratorOutput(AbstractIsingSubproblem):
    """
    This constraint enforces the minimal output of a generator by transforming
    positive first-order interactions of generator qubits into a second-order
    interaction with a status qubit
    """

    @classmethod
    def build_subproblem(cls,
                         backbone: IsingBackbone,
                         configuration: dict):
        """
        A factory method for returning an instance that enforces the minimal
        generator output.

        Args:
            backbone: (IsingBackbone)
                The ising_backbone which to modify
            configuration: (dict)
                A stub for the config dictionary that is passed, but this
                argument is ignored

        Returns:
            (MinimalGeneratorOutput)
                An instance of this class
        """
        return MinimalGeneratorOutput(backbone, configuration)

    def encode_subproblem(self) -> None:
        """
        Modifies the first order interactions of generator qubits into
        second-oder interactions with a status qubit

        Returns:
            (None)
                Modifies `self.backbone.ising_coefficients` and
                `self.backbone.ising_coefficients_positive
        """
        for generator in self.network.generators.index:
            self.modifiy_positive_interactions(generator=generator)
        self.ising_coefficients = self.backbone.ising_coefficients_cached

    def modifiy_positive_interactions(self, generator: str):
        """
        Modifies the first-order ising interactions of the generator
        at the given time slice

        Args:
            generator: (str) label of the network generator

        Returns:
            (None)
                Modifies the attribute `self.backbone`
        """
        generator_qubit_map = self.backbone._qubit_encoding[generator]
        for qubit_list in generator_qubit_map.values():
            if not qubit_list:
                continue
            status_qubit = qubit_list[0]
            for qubit in qubit_list[1:]:
                interaction_strength = abs(self.scale_factor * self.backbone.ising_coefficients[(qubit,)])
                self.backbone.add_basis_polynomial_interaction(first_qubit=status_qubit,
                                                               second_qubit=qubit,
                                                               zero_qubits_list=[status_qubit],
                                                               interaction_strength=interaction_strength)

class PowerOutputInvariant(AbstractIsingSubproblem):
    """
    This constraint enforces that the total power output is equal to the total
    load in the network. Due to the kirchhoff constraint being a local, bus-based
    constraint, the implicit penalty on it is quasi linear. For bad marginal cost
    estimations, this leads to very bad solutions. This also increases the performance
    of the marginal cost encoding by explicitly adding the invariant it's encoding
    relies on to be penalized quadratically
    """

    @classmethod
    def build_subproblem(cls,
                         backbone: IsingBackbone,
                         configuration: dict):
        """
        A factory method for returning an instance that enforces the minimal
        generator output.

        Args:
            backbone: (IsingBackbone)
                The ising_backbone which to modify
            configuration: (dict)
                The config dict that contains the scale factor

        Returns:
            (MinimalGeneratorOutput)
                An instance of this class
        """
        return PowerOutputInvariant(backbone, configuration)

    def encode_subproblem(self) -> None:
        """
        Modifies the first order interactions of generator qubits into
        second-oder interactions with a status qubit

        Returns:
            (None)
                Modifies `self.backbone.ising_coefficients` and
                `self.backbone.ising_coefficients_positive
        """
        for time in self.backbone.snapshots:
            self.encode_total_power_invariant(time=time)

    def encode_total_power_invariant(self, time=None):
        """
        Encodes that the total generated power is equal to the total load
        at that time step

        Args:
            time: (any)
                The snapshot at which to encode the constraint

        Returns:
            (None)
                Modifies the attribute `self.backbone`
        """
        if time is None:
            time = self.backbone.snapshots[0]
        self.backbone.encode_squared_distance(
            label_list=self.backbone.network.generators.index,
            target=-self.backbone.get_total_load(time),
            global_factor=self.scale_factor,
            time=time,
        )
