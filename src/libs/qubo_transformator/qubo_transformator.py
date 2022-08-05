import pypsa

from .ising_backbone import IsingBackbone
from .qubit_encoder import GeneratorEncoder, LineEncoder
from .ising_subproblems import KirchhoffSubproblem, MarginalCostSubproblem, MinimalGeneratorOutput

import typing

class QuboTransformator:

    def __init__(self, config):
        pass    


    def transform_network_to_qubo(self) -> IsingBackbone:
        result = IsingBackbone()
        # TODO call functions

        return result

