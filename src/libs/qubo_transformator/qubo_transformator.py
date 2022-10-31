import pypsa

from .ising_backbone import IsingBackbone, NetworkIsingBackbone
from .qubit_encoder import NetworkEncoder
from .ising_subproblems import KirchhoffSubproblem, MarginalCostSubproblem, MinimalGeneratorOutput

import typing


class TspTransformator:

    def __init__(self, graph_dict, config):
        self.graph = graph
        self.config = config

    def transform_network_to_qubo(self) -> IsingBackbone:
        backbone_result = IsingBackbone()
        print()
        print("--- Generating Ising problem ---")

class QuboTransformator:

    def __init__(self, network, config):
        self.network = network
        self.config = config

    def transform_network_to_qubo(self) -> NetworkIsingBackbone:
        backbone_result = NetworkIsingBackbone(self.network)
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
        NetworkEncoder.create_encoder(backbone_result, self.config).encode_qubits()
        for subproblem, subproblem_configuration in self.config.items():
            if subproblem in ["generator_representation", "line_representation"]:
                continue
            if subproblem not in self.subproblem_table:
                print(f"{subproblem} is not a valid subproblem, skipping "
                      f"encoding")
                continue
            if subproblem_configuration is None:
                subproblem_configuration = {}
            subproblem_instance = self.subproblem_table[
                subproblem].build_subproblem(backbone_result, subproblem_configuration)
            backbone_result._subproblems[subproblem] = subproblem_instance
            backbone_result.flush_cached_problem()
            subproblem_instance.encode_subproblem()
        print()
        print("--- Finish generating Ising Problem with the following subproblems ---")
        for key in backbone_result._subproblems:
            print("--- - " + key)
        return backbone_result

