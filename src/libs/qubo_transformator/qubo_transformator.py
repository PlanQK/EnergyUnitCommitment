import pypsa

from .ising_backbone import IsingBackbone, NetworkIsingBackbone
from .qubit_encoder import NetworkEncoder
from .ising_subproblems import KirchhoffSubproblem, MarginalCostSubproblem, MinimalGeneratorOutput, PowerOutputInvariant

import typing


class TspTransformator:

    def __init__(self, graph_dict, config):
        self.graph = graph_dict
        self.nodes = self.get_nodes()
        self.edges = [str(edge) for edge in self.graph]
        self.config = config

    def get_nodes(self):
        nodes = []
        for edge in self.graph:
            nodes += edge
        return list(set(nodes))

    def get_adjacent_edges(self, node):
        return [str(edge) for edge in self.graph if node in edge]

    def transform_network_to_qubo(self) -> IsingBackbone:
        backbone_result = IsingBackbone()
        print()
        print("--- Generating Ising problem ---")
        GraphEncoder.create_encoder(backbone_result, self.graph).encode_qubits()
        backbone_result.encode_squared_distance(
                label_list=self.edges,
                target=-len(self.nodes))
        for node in self.nodes:
            adj_edges = self.get_adjacent_edges(node)
            backbone_result.encode_squared_distance(
                    label_list=adj_edges,
                    target=-2)
        # weight function
        target = -4.0
        backbone_result.encode_squared_distance(
                label_dictionary={str(edge) : weight for edge, weight in self.graph.items()},
                target=target,
                )
        return backbone_result

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
            "minimal_power": MinimalGeneratorOutput,
            "total_power": PowerOutputInvariant
        }
        if "kirchhoff" not in self.config:
            print("No Kirchhoff configuration found, "
                  "adding Kirchhoff constraint with Factor 1.0")
            self.config["kirchhoff"] = {"scale_factor": 1.0}

        # encode the network components into qubits
        NetworkEncoder.create_encoder(backbone_result, self.config).encode_qubits()
        unmatched_subproblems = []
        for subproblem, subproblem_configuration in self.config.items():
            if subproblem in ["generator_representation", "line_representation"]:
                continue
            if subproblem not in self.subproblem_table:
                print(f"{subproblem} is not a valid subproblem, deleting and skipping "
                      f"encoding")
                unmatched_subproblems.append(subproblem)
                continue
            if subproblem_configuration is None:
                subproblem_configuration = {}
            subproblem_instance = self.subproblem_table[
                subproblem].build_subproblem(backbone_result, subproblem_configuration)
            backbone_result._subproblems[subproblem] = subproblem_instance
            backbone_result.flush_cached_problem()
            subproblem_instance.encode_subproblem()
        for subproblem in unmatched_subproblems:
            self.config.pop(subproblem)
        print()
        print("--- Finish generating Ising Problem with the following subproblems ---")
        for key in backbone_result._subproblems:
            print("--- - " + key)
        return backbone_result

