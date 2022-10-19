import pytest
import sys

import pypsa

from src.libs.qubo_transformator.qubit_encoder import GeneratorEncoder
from src.libs.qubo_transformator.ising_backbone import IsingBackbone

from .pypsa_networks import create_network

@pytest.fixture(scope='session')
def config():
    return {
        "backend": "sqa",
        "ising_interface": {
            "generator_representation": "single_qubit",
            "line_representation": "fullsplit",
            "kirchhoff": {
                "scale_factor": 1.0
            },
            "marginal_cost": {
                "scale_factor": 0.5,
                "strategy": "global_cost_square",
                "target_cost": 0.0,
                "range_factor": 0.0,
                "offset": 0.0,
            }
        },
        "sqa_backend": {
            "trotter_slices": 20,
            "optimization_cycles": 5,
        }
    }


@pytest.fixture
def backbone(network):
    return IsingBackbone(network)

@pytest.fixture
def generator_rep():
    return "single_qubit"

@pytest.fixture
def line_rep():
    return "fullsplit"

@pytest.mark.parametrize("network_loads", [
    [3],
    [3,4],
    [3,4,2],
])
def test_generator_init(network_loads, generator_rep):
    network = create_network(network_loads)
    backbone = IsingBackbone(network)
    snapshot_count = len(network_loads)
    encoder = GeneratorEncoder.create_encoder(backbone, generator_rep)
    assert encoder.backbone == backbone, "encoder references wrong backbone"
    assert encoder.network == backbone.network, "encoder references wrong network"

    encoder.encode_qubits()
    for snapshot in network.snapshots:
        print(backbone.get_representing_qubits("gen_1", snapshot), file=sys.stderr)
        assert backbone.get_representing_qubits("gen_1", snapshot) == [snapshot]
        assert backbone.get_representing_qubits("gen_2", snapshot) == [snapshot_count + snapshot]
        assert backbone._qubit_weights[snapshot] == 4
        assert backbone._qubit_weights[snapshot_count + snapshot] == 3
