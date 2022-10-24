import pytest
import typing

import pypsa

from src.libs.qubo_transformator.ising_backbone import IsingBackbone

# creates a networj with 2 generators at two buses. The first "gen_1" has a power output
# of 4 and the second "gen_2" has a power output of 3
from .pypsa_networks import create_network

from src.libs.qubo_transformator.qubit_encoder import GeneratorEncoder

@pytest.fixture
def backbone():
    """
    Create an ising backbone, that only has the generators encodes as qubits
    """
    backbone_result = IsingBackbone(create_network([3]))
    encoder = GeneratorEncoder.create_encoder(backbone_result, "single_qubit")
    encoder.encode_qubits()
    return backbone_result

def test_backbone_init(backbone):
    """
    Check that the qubit-network encoding data in the backbone is correct
    """
    assert backbone.ising_coefficients == {}
    assert backbone.get_representing_qubits("gen_1") == [0]
    assert backbone.get_representing_qubits("gen_2") == [1]
    assert backbone.get_nominal_power("gen_1", time=0) == 4
    assert backbone.get_nominal_power("gen_2", time=0) == 3


def test_add_interaction(backbone):
    """
    Test adding interactions with different arguments and signs. The sign 
    in the interaction for adding positive weights is a due to a convention
    """
    backbone.add_interaction(0, 1.0)
    assert backbone.ising_coefficients == {(0,): -4.0}
    backbone.add_interaction(1,2.0)
    assert backbone.ising_coefficients == {(0,): -4.0, (1,): -6.0, }
    backbone.add_interaction(0, 1, 1.0)
    assert backbone.ising_coefficients == {(0,): -4.0, (1,): -6.0, (0,1): -12.0}
    backbone.add_interaction(0, -2.0)
    assert backbone.ising_coefficients == {(0,): 4.0, (1,): -6.0, (0,1): -12.0}
    backbone.add_interaction(0, 1, -0.5)
    assert backbone.ising_coefficients == {(0,): 4.0, (1,): -6.0, (0,1): -6.0}
    backbone.add_interaction(0, 0, 1.0)
    assert backbone.ising_coefficients == {():  -16.0, (0,): 4.0, (1,): -6.0, (0,1): -6.0}
    backbone.add_interaction(0, 0.0)
    assert backbone.ising_coefficients == {():  -16.0, (0,): 4.0, (1,): -6.0, (0,1): -6.0}
    backbone.add_interaction(0,1, 0.0)
    assert backbone.ising_coefficients == {():  -16.0, (0,): 4.0, (1,): -6.0, (0,1): -6.0}
    backbone.add_interaction(1, 0.0)
    assert backbone.ising_coefficients == {():  -16.0, (0,): 4.0, (1,): -6.0, (0,1): -6.0}
    backbone.add_interaction(1, 0, -1.0, weighted_interaction=False)
    assert backbone.ising_coefficients == {():  -16.0, (0,): 4.0, (1,): -6.0, (0,1): -5.0}


def test_couple_components(backbone):
    """
    Test that coupling network components by their labels gets resolved correctly
    """
    backbone.couple_components("gen_1", "gen_2", coupling_strength=2.0)
    assert backbone.ising_coefficients == {(): -6.0, (0,): -6.0, (0, 1): -6.0, (1,): -6.0}


def test_couple_component_with_constant(backbone):
    """
    Test that adding a first order interaction to all qubits of a network component works
    """
    backbone.couple_component_with_constant("gen_1", coupling_strength=1.0)
    # assert backbone.ising_coefficients == {(): -2.0, (0,): -2.0}
    assert backbone.calc_cost([]) == 0
    assert backbone.calc_cost([0]) == 4
    backbone.couple_component_with_constant("gen_2", coupling_strength=-2.0)
    assert backbone.calc_cost([]) == 0
    assert backbone.calc_cost([1]) == -6
    assert backbone.calc_cost([0, 1]) == -2

@pytest.mark.parametrize("penalized_solution", [
    [],
    [0],
    [1],
    [0,1],
])
def test_add_basis_polynomial_interaction(penalized_solution, backbone):
    """
    The added interaction should work in a way, that the cost function hasn't been
    changed except for the case that the qubits of the first two arguments are in
    state as given by the solution. The solution is given the same way that the sqa
    solver would return the state
    """
    backbone.add_basis_polynomial_interaction(0, 1, penalized_solution, 2.0)
    states =[[],[0],[1],[0,1]]
    for state in states:
        if state == penalized_solution:
            assert backbone.calc_cost(state) == 2.0
        else:
            assert backbone.calc_cost(state) == 0.0

def test_encode_squared_distance(backbone):
    """
    Test encoding the squared distance to a target by a list of network labels
    """
    label_dictionary={"gen_1": 1.0, "gen_2": 1.0}
    target = 0.0
    backbone.encode_squared_distance(label_dictionary,
                                    global_factor=1.0,
                                    target = 0.0)
    assert backbone.ising_coefficients == {(): -18.5, (0,): -14.0, (0, 1): -6.0, (1,): -10.5}
    assert backbone.calc_cost([]) == 0.0
    assert backbone.calc_cost([0]) == 16.0
    assert backbone.calc_cost([1]) == 9.0
    assert backbone.calc_cost([0,1]) == 49.0
    backbone.ising_coefficients = {}
    backbone.encode_squared_distance(label_dictionary,
                                    global_factor=1.0,
                                    target = -3.0)
    assert backbone.calc_cost([]) == 9.0
    # assert backbone.calc_cost([0]) == 1.0
    assert backbone.calc_cost([1]) == 0.0
    assert backbone.calc_cost([0, 1]) == 16.0
    backbone.ising_coefficients = {}
    backbone.encode_squared_distance(label_dictionary,
                                    global_factor=1.0,
                                    target = 3.0)
    assert backbone.calc_cost([]) == 9.0
    assert backbone.calc_cost([0]) == 49
    assert backbone.calc_cost([1]) == 36
    assert backbone.calc_cost([0, 1]) == 100
