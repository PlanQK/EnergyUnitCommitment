"""
This module contains tests concerning the functionality of the GlobalCostSquare
class derived from IsingSubproblem.
"""
import typing
import pytest
import pypsa

from src.libs.qubo_transformator.ising_backbone import NetworkIsingBackbone

from src.libs.qubo_transformator.ising_subproblems import GlobalCostSquare

from src.libs.qubo_transformator.qubit_encoder import GeneratorEncoder

# creates a network with 2 generators at two buses. The first "gen_1" has a power output
# of 4 and the second "gen_2" has a power output of 3
from .pypsa_networks import create_network


@pytest.fixture
def create_backbone():
    """
    Create an ising backbone, that only has the generators encodes as qubits
    """    
    backbone_result = NetworkIsingBackbone(create_network([3]))
    encoder = GeneratorEncoder.create_encoder(backbone_result, "single_qubit")
    encoder.encode_qubits()
    return backbone_result

def test_network_cost(backbone):
    """
    Test that the loaded network in the backbone has the correct values for the
    following tests
    """
    network = create_network([3])
    assert network.generators.marginal_cost.loc["gen_1"] == 15
    assert network.generators.marginal_cost.loc["gen_2"] == 10
    assert backbone.get_nominal_power("gen_1", time=0) == 4
    assert backbone.get_nominal_power("gen_2", time=0) == 3

def test_global_cost_square_distance_encoding_by_offset(backbone):
    """
    Test encoding the marginal costs by giving a generator cost offset, which
    implcitly sets a target as the product of the offset and the total load
    """
    config = {"strategy": "global_cost_square",
              "offset": 10,
              }
    cost_encoder = GlobalCostSquare.build_subproblem(backbone, config)
    cost_encoder.encode_subproblem()
    assert backbone.calc_cost([]) == 0.0
    assert backbone.calc_cost([0]) == 400
    assert backbone.calc_cost([1]) == 0.0
    assert backbone.calc_cost([0, 1]) == 400

def test_global_cost_square_distance_encoding_min_gen_offset(backbone):
    """
    Test that using the target_cost overrides the offset configuration key
    """
    config = {"strategy": "global_cost_square",
              "target_cost": 30,
              "offset": 0,
              }
    cost_encoder = GlobalCostSquare.build_subproblem(backbone, config)
    cost_encoder.encode_subproblem()
    assert backbone.calc_cost([]) == 0
    assert backbone.calc_cost([0]) == 400
    assert backbone.calc_cost([1]) == 0
    assert backbone.calc_cost([0, 1]) == 400


def test_global_cost_square_distance_encoding_zero_target_offset(backbone):
    """
    Test encoding the marginal cost by giving a target value of the cost
    by checking the cost of all possible states and the ising coefficients
    """
    config = {"strategy": "global_cost_square",
              "target_cost": 40,
              }
    cost_encoder = GlobalCostSquare.build_subproblem(backbone, config)
    cost_encoder.encode_subproblem()
    assert backbone.get_ising_coefficients() == {
            (): -38.88888888888889,
            (0,): 11.111111111111121,
            (0, 1): 33.33333333333333,
            (1,): -16.666666666666686
    }
    assert backbone.calc_cost([]) == 0.0
    assert backbone.calc_cost([0]) == 44.444444444444414
    assert backbone.calc_cost([1]) == 100.00000000000003
    assert backbone.calc_cost([0, 1]) == 11.111111111111128
