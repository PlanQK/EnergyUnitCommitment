"""The classes defined in this file consume a configuration and a problem instance of
an optimization problem, and construct the finished ising model as an IsingBackbone.
This is done by initializing an appropriate IsingBackbone subclassinstace, create and
encode the problem components via a QubitEncoder, and then using IsingSubproblem instances
to encode the various constraints and the objective function.
"""
import pypsa


from .ising_backbone import NetworkIsingBackbone
from .qubit_encoder import NetworkEncoder
from .ising_subproblems import KirchhoffSubproblem, MarginalCostSubproblem, MinimalGeneratorOutput, PowerOutputInvariant


class QuboTransformator:
    """
    This class is a collection of transformation methods for transformation various
    optimization problems into QUBO form
    """
    @classmethod
    def transform_network_to_qubo(cls,
                                  network: pypsa.Network,
                                  config: dict
                                  ) -> NetworkIsingBackbone:
        """
        Transforms the unit commitment problem of a pypsa.Network into a QUBO

        Args:
            network: (pypsa.Network)
                The network that contains the unit commitment problem 
            config: (dict)
                A configuration that describes how generators and transmission
                lines are encoded and which constraints are added to the QUBO
        """
        backbone_result = NetworkIsingBackbone(network)
        print()
        print("--- Generating Ising problem ---")
        subproblem_table = {
            "kirchhoff": KirchhoffSubproblem,
            "marginal_cost": MarginalCostSubproblem,
            "minimal_power": MinimalGeneratorOutput,
            "total_power": PowerOutputInvariant
        }
        if "kirchhoff" not in config:
            print("No Kirchhoff configuration found, "
                  "adding Kirchhoff constraint with Factor 1.0")
            config["kirchhoff"] = {"scale_factor": 1.0}

        # encode the network components into qubits
        NetworkEncoder.create_encoder(backbone_result, config).encode_qubits()
        unmatched_subproblems = []
        for subproblem, subproblem_configuration in config.items():
            if subproblem in ["generator_representation", "line_representation"]:
                continue
            if subproblem not in subproblem_table:
                print(f"{subproblem} is not a valid subproblem, deleting and skipping "
                      f"encoding")
                unmatched_subproblems.append(subproblem)
                continue
            if subproblem_configuration is None:
                subproblem_configuration = {}
            subproblem_instance = subproblem_table[
                subproblem].build_subproblem(backbone_result, subproblem_configuration)
            backbone_result._subproblems[subproblem] = subproblem_instance
            backbone_result.flush_cached_problem()
            subproblem_instance.encode_subproblem()
        for subproblem in unmatched_subproblems:
            config.pop(subproblem)
        print()
        print("--- Finish generating Ising Problem with the following subproblems ---")
        for key in backbone_result._subproblems:
            print("--- - " + key)
        return backbone_result
