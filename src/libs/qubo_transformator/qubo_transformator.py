import pypsa

from .ising_backbone import IsingBackbone
from .qubit_encoder import GeneratorEncoder, LineEncoder
from .ising_subproblems import KirchhoffSubproblem, MarginalCostSubproblem, MinimalGeneratorOutput

import typing


class QuboTransformator:

    def __init__(self, network, config):
        self.network = network
        self.config = config

    def transform_network_to_qubo(self) -> IsingBackbone:
        result = IsingBackbone(self.network, self.config)

        print()
        print("--- Generating Ising problem ---")
        self.subproblem_table = {
            "kirchhoff": KirchhoffSubproblem,
            "marginal_cost": MarginalCostSubproblem,
            "minimal_power": MinimalGeneratorOutput
        }
        if "kirchhoff" not in self.config:
            print("No Kirchhoff configuration found, "
                  "adding Kirchhoff constraint with Factor 1.0")
            self.config["kirchhoff"] = {"scale_factor": 1.0}

        # encode the network components into qubits
        generator_representation = self.config.pop("generator_representation", "single_qubit")
        line_representation = self.config.pop("line_representation", "cutpowersoftwo")
        GeneratorEncoder.create_encoder(result,
                                        generator_representation).encode_qubits()
        LineEncoder.create_encoder(result, 
                                   line_representation).encode_qubits()

        for subproblem, subproblem_configuration in self.config.items():
            if subproblem not in self.subproblem_table:
                print(f"{subproblem} is not a valid subproblem, skipping "
                      f"encoding")
                continue
            if subproblem_configuration is None:
                subproblem_configuration = {}
            subproblem_instance = self.subproblem_table[
                subproblem].build_subproblem(result, subproblem_configuration)
            result._subproblems[subproblem] = subproblem_instance
            result.flush_cached_problem()
            subproblem_instance.encode_subproblem()
            # print(subproblem)
            # print(result.ising_coefficients_cached)
        print()
        print("--- Finish generating Ising Problem with the following subproblems ---")
        for key in result._subproblems:
            print("--- - " + key)
        # readd representation values to config for the result dict containing that info
        self.config["generator_representation"] = generator_representation
        self.config["line_representation"] = line_representation
        # print(result.get_hamiltonian_matrix())
        # raise ValueError

        return result
