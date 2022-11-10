from abc import ABC, abstractmethod
from typing import Union


from .ising_backbone import IsingBackbone, NetworkIsingBackbone


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


class QubitEncoder(ABC):
    """
    An interface for objects that transform network components
    into qubit representation and encode this into the backbone

    For each type of component that is encoded into qubits, we 
    implement another interface that extends this and whose 
    subclasses transform that type of component.
    """

    def __init__(self, backbone: IsingBackbone):
        """
        The only reference a generic encoder requires is to a
        backbone instance on which to encode components as qubits

        Args:
            backbone: (NetworkIsingBackbone)
                The backbone which is used to encode qubits
        """
        self.backbone = backbone

    @classmethod
    @abstractmethod
    def create_encoder(cls,
                       backbone: IsingBackbone,
                       config: Union[str, dict]
                       ) -> 'QubitEncoder':
        """
        A factory method to be overwritten by the interface that
        extends this one. This method will be called to get the 
        encoder for one type of network component

        Args:
            backbone: (IsingBackbone)
                The IsingBackbone instance on which to encode the 
                network component
            config: (str)
                A string describing which subclass to instantiate
                for generator encoding
        """

    def encode_qubits(self) -> None:
        """
        The entrypoint for the backbone to call to encode the network
        component that is specific to the class of the instance.
        """
        for component in self.get_components():
            self.backbone.create_qubit_entries_for_component(
                component_name=component,
                snapshot_to_weight=self.encoding_method(component)
                )

    def encoding_method(self, component: str) -> dict:
        """
        Returns the dictionary with all weights of the component
        for all time steps
        """
        return {
                time: self.get_weights(component, time)
                for time in self.backbone._snapshots
                }

    def get_weights(self, component: str, time: any) -> list:
        """
        For the given component and snapshot, this methods returns the
        weights of qubit that encode it. The component type is
        dependent of the subclass of the encoder
        """
        raise NotImplementedError

    def get_components(self):
        """
        Returns an iterable of all components this instances transforms into
        qubits.
        """
        raise NotImplementedError


class GeneratorEncoder(QubitEncoder, ABC):
    """
    An encoder for transforming generators into groups of qubits.
    """

    def get_components(self) -> list:
        """
        Returns a list of all network generators
        """
        return list(self.backbone.network.generators.index)

    @classmethod
    def create_encoder(cls,
                       backbone: NetworkIsingBackbone,
                       config: str
                       ) -> 'GeneratorEncoder':
        """
        A factory method for constructing an encoder for generators

        Admissible string for the config are:
            - `"single_qubit"`
            - `"integer_decomposition"`
            - `"with_status"`

        Args:
            backbone: (IsingBackbone)
                The IsingBackbone instance on which to encode the 
                network component
            config: (str)
                A string describing which subclass to instantiate
                for generator encoding
        """
        if config == "single_qubit":
            return SingleQubitGeneratorEncoder(backbone)
        elif config == "integer_decomposition":
            return BinaryPowerGeneratorEncoder(backbone)
        elif config == "with_status":
            return WithStatusQubitGeneratorEncoder(backbone)
        raise ValueError(f"{config} is not a valid option for generator encoding")


class SingleQubitGeneratorEncoder(GeneratorEncoder):
    """
    Transform generators into a single qubit with weight equal to
    the power output of the generator
    """

    def get_weights(self, component: str, time: any) -> list:
        """
        Returns a list with its only entry being the maximal powerout
        of the generator at the time step

        Args:
            component: (str)
                The label of the generator for which to calculate the weight
            time: (any)
                the snapshot for which to calculate the weight for

        Returns:
            (list) 
                A list of positive numbers that sum up to the maximal
                power output of the generator
        """
        return [int(self.backbone.get_nominal_power(component, time))]


class BinaryPowerGeneratorEncoder(GeneratorEncoder):
    """
    Transform generators into qubits with powers of two and a rest as weights
    """

    def get_weights(self, component: str, time: any) -> list:
        """
        Returns the binary decomposition and the rest for the maximal 
        output of the component

        Args:
            component: (str)
                The label of the generator for which to calculate the weight
            time: (any)
                the snapshot for which to calculate the weight for

        Returns:
            (list) 
                A list of positive numbers that sum up to the maximal
                power output of the generator
        """
        return binary_power_and_rest(int(self.backbone.get_nominal_power(component, time)))


class WithStatusQubitGeneratorEncoder(GeneratorEncoder):
    """
    Transforms a generator by using a qubit with weight equal to the minimal
    power and filling the range to the maximal power using binary powers
    a rest.

    This generator encoding is compatible with the MinimalGeneratorOutput constraint
    because it is sets the first weight at each time step to the minimal poweroutput
    that is to be enforced
    """

    def get_weights(self, component: str, time: any) -> list:
        """
        Returns a list with the first entry as the minimal output of
        the generator and the range up to the maximum output being
        filled using binary powers and a rest.

        Args:
            component: (str)
                The label of the generator for which to calculate the weight
            time: (any)
                the snapshot for which to calculate the weight for

        Returns:
            (list) 
                A list of positive numbers that sum up to the maximal
                power output of the generator and the first entry being\
                the minimal output of the generator
        """
        minimal_power = self.backbone.get_minimal_power(component, time)
        max_power = self.backbone.get_nominal_power(component, time)
        return [minimal_power] + binary_power_and_rest(max_power - minimal_power)


class LineEncoder(QubitEncoder, ABC):
    """
    An encoder for transforming transmission lines into groups of qubits.
    """

    def get_components(self) -> list:
        """
        Returns a list of all network transmission lines
        """
        return list(self.backbone.network.lines.index)

    @classmethod
    def create_encoder(cls, 
                       backbone: NetworkIsingBackbone,
                       config: str
                       ) -> 'LineEncoder':
        """
        A factory method for constructing an encoder for generators

        Admissible string for the config are:
            - `"single_qubit"`
            - `"integer_decomposition"`

        Args:
            backbone: (NetworkIsingBackbone)
                The NetworkIsingBackbone instance on which to encode the 
                network component
            config: (str)
                A string describing which subclass to instantiate
                for generator encoding
        """
        if config == "fullsplit":
            return FullsplitLineEncoder(backbone)
        elif config == "cutpowersoftwo":
            return CutPowersOfTwoLineEncoder(backbone)
        raise ValueError(f"{config} is not a valid option for generator encoding")


class FullsplitLineEncoder(LineEncoder):
    """
    Transform a transmission line by using only qubits of weight 1 or -1
    """

    def get_weights(self, component: str, time: any) -> list:
        """
        Split up a transmission line using the smallest integer-valued weights
        as possible

        Args:
            component: (str)
                The label of the transmission line for which to calculate the weight
            time: (any)
                the snapshot for which to calculate the weight for

        Returns:
            (list) 
                Returns a list of 1 and -1 with each number occurring equal
                to the capacity of the transmission line at that timestep
        """
        capacity = int(self.backbone.network.lines.loc[component].s_nom)
        return capacity * [1] + capacity * [-1]


class CutPowersOfTwoLineEncoder(LineEncoder):
    """
    Transforms a transmission line by using powers of two and a rest term for
    each direction of the flow
    """

    def get_weights(self, component: str, time: any) -> list:
        """
        Split up a transmission line using powers of two and a rest for
        each direction

        Args:
            component: (str)
                The label of the transmission line for which to calculate the weight
            time: (any)
                the snapshot for which to calculate the weight for

        Returns:
            (list) 
                A list of powers of two and a rest term with positive and
                negative sign
        """
        capacity = int(self.backbone.network.lines.loc[component].s_nom)
        return self.cut_powers_of_two(capacity)

    def cut_powers_of_two(self, capacity: float) -> list:
        """
        A method for splitting up the capacity of a line with a given
        maximum capacity.

        It uses powers of two to decompose the capacity and cuts off
        the biggest power of two so the total sum of all powers equals
        the capacity. The capacity is also rounded to an integer.
        
        Args:
            capacity: (int)
                The capacity of the line to be decomposed.
        Returns:
            (list)
                A list of weights to be used in decomposing a line.
        """
        integer_capacity = int(capacity)
        positive_capacity = binary_power_and_rest(integer_capacity)
        negative_capacity = [- number for number in positive_capacity]
        return positive_capacity + negative_capacity


class NetworkEncoder(QubitEncoder):
    """
    A class for transforming the components of a pypsa Network into qubits.
    """

    def __init__(self,
                 backbone: NetworkIsingBackbone,
                 generator_encoder: GeneratorEncoder,
                 line_encoder: LineEncoder):
        """
        Args:
           backbone: (NetworkIsingBackbone)
               The backbone that organizes qubits that represent network components
           generator_encoder: (GeneratorEncoder)
               A qubit encoder that reads the network generators and encodes
               them as qubits into the backbone
           line_encoder: (LineEncoder)
               A qubit encoder that reads the transmission lines and encodes
               them as qubits into the backbone
        """
        super().__init__(backbone)
        self.generator_encoder = generator_encoder
        self.line_encoder = line_encoder

    @classmethod
    def create_encoder(cls, 
                       backbone: NetworkIsingBackbone,
                       config: dict
                       ) -> 'NetworkEncoder':
        """
        This creates a NetworkEncoder by creating the network and generator
        encoders as given in the config. Encoding the network is archived
        by applying the encoding methods of those

        Args:
            backbone: (NetworkIsingBackbone)
                The NetworkIsingBackbone instance on which to encode the network
            config: (dict)
                A dictionary containing the config for the generator and
                transmission line encoding
        """
        generator_representation = config.get("generator_representation", "single_qubit")
        line_representation = config.get("line_representation", "cutpowersoftwo")
        generator_encoder = GeneratorEncoder.create_encoder(backbone,
                                                            generator_representation)
        line_encoder = LineEncoder.create_encoder(backbone,
                                                  line_representation)
        return NetworkEncoder(backbone, generator_encoder, line_encoder)

    def encode_qubits(self) -> None:
        """
        The entrypoint for the backbone to call to encode the network
        """
        self.generator_encoder.encode_qubits()
        self.line_encoder.encode_qubits()
