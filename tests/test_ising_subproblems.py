import pytest
import typing

import pypsa

from src.libs.qubo_transformator.ising_backbone import IsingBackbone

from src.libs.qubo_transformator.ising_subproblems import GlobalCostSquare

# creates a network with 2 generators at two buses. The first "gen_1" has a power output
# of 4 and the second "gen_2" has a power output of 3
from .pypsa_networks import create_network

from src.libs.qubo_transformator.qubit_encoder import GeneratorEncoder

@pytest.fixture
def backbone():
    backbone_result = IsingBackbone(create_network([3]), {})
    encoder = GeneratorEncoder.create_encoder(backbone_result, "single_qubit")
    encoder.encode_qubits()
    return backbone_result

def test_network_cost(backbone):
    network = create_network([3])
    assert network.generators.marginal_cost.loc["gen_1"] == 15
    assert network.generators.marginal_cost.loc["gen_2"] == 10
    assert backbone.get_nominal_power("gen_1", time=0) == 4
    assert backbone.get_nominal_power("gen_2", time=0) == 3

def test_global_cost_square_distance_encoding_no_offset(backbone):
    config = {"strategy": "global_cost_square",
              "target_cost": 40,
              "offset": -10,
              }
    cost_encoder = GlobalCostSquare.build_subproblem(backbone, config)
    cost_encoder.encode_subproblem()
    assert backbone.calc_cost([]) == 1600
    assert backbone.calc_cost([0]) == 400
    assert backbone.calc_cost([1]) == 100
    assert backbone.calc_cost([0, 1]) == 2500

def test_global_cost_square_distance_encoding_min_gen_offset(backbone):
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
    config = {"strategy": "global_cost_square",
              "target_cost": 40,
              }
    cost_encoder = GlobalCostSquare.build_subproblem(backbone, config)
    cost_encoder.encode_subproblem()
    assert backbone.ising_coefficients == {
            (): -38.88888888888889,
            (0,): 11.111111111111121,
            (0, 1): 33.33333333333333,
            (1,): -16.666666666666686
    }
    assert backbone.calc_cost([]) == 0.0
    assert backbone.calc_cost([0]) == 44.444444444444414
    assert backbone.calc_cost([1]) == 100.00000000000003
    assert backbone.calc_cost([0, 1]) == 11.111111111111128
