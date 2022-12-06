"""
This module contains a test concerning the functionality of the GeneratorEncoder
and more specifically SingleQubitGeneratorEncoder classes, derived from QubitEncoder.
"""
import sys
import pytest
import pypsa

from src.libs.qubo_transformator.qubit_encoder import GeneratorEncoder
from src.libs.qubo_transformator.ising_backbone import NetworkIsingBackbone

from .pypsa_networks import create_network


@pytest.fixture(scope="session")
def config():
    """
    Set configuration parameters.
    """
    return {
        "backend": "sqa",
        "ising_interface": {
            "generator_representation": "single_qubit",
            "line_representation": "fullsplit",
            "kirchhoff": {"scale_factor": 1.0},
            "marginal_cost": {
                "scale_factor": 0.5,
                "strategy": "global_cost_square",
                "target_cost": 0.0,
                "range_factor": 0.0,
                "offset": 0.0,
            },
        },
        "sqa_backend": {
            "trotter_slices": 20,
            "optimization_cycles": 5,
        },
    }


@pytest.fixture
def generator_rep():
    """
    Returns configuration string for single qubit encoding.
    """
    return "single_qubit"


@pytest.mark.parametrize(
    "network_loads",
    [
        [3],
        [3, 4],
        [3, 4, 2],
    ],
)
def test_generator_init(network_loads, generator_rep):
    """
    Check that for various amount of snapshots, the generators get encoded correctly
    into the backbone as qubits
    """
    network = create_network(network_loads)
    backbone = NetworkIsingBackbone(network)
    snapshot_count = len(network_loads)
    encoder = GeneratorEncoder.create_encoder(backbone, generator_rep)
    # check references in the backbone
    assert encoder.backbone == backbone, "encoder references wrong backbone"
    # check that each value in each snapshots is encoded
    encoder.encode_qubits()
    for snapshot in network.snapshots:
        assert backbone.get_representing_qubits("gen_1", snapshot) == [snapshot]
        assert backbone.get_representing_qubits("gen_2", snapshot) == [
            snapshot_count + snapshot
        ]
        assert backbone.get_qubit_weights()[snapshot] == 4
        assert backbone.get_qubit_weights()[snapshot_count + snapshot] == 3
