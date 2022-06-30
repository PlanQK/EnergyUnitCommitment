import pytest
import typing

import sys
sys.path.append('../src')

import pypsa

import networkx as nx

from libs.Backends.IsingPypsaInterface import IsingBackbone


@pytest.fixture(scope='session')
def network_path():
    network_path = "../sweepNetworks/testNetwork4QubitIsing_2_0_20.nc"
    network_path = "../sweepNetworks/elec_s_5.nc"
    # network_path = "../sweepNetworks/20220629_network_5_0_20.nc"
    # network_path = "../sweepNetworks/20220629_network_3_0_20.nc"
    return network_path


@pytest.fixture(scope='session')
def num_snapshots():
    return 3


@pytest.fixture(scope='session')
def network(network_path, num_snapshots):
    network = pypsa.Network(network_path)
    network.snapshots = network.snapshots[:num_snapshots]
    return network


@pytest.fixture
def config():
    config = {
        "backend": "sqa",
        "ising_interface": {"formulation": "cutpowersoftwo"},
        "snapshots": 4,
        "sqa_backend": {
            "transverse_field_schedule": "[8.0,0.0]",
            "temperature_schedule": "[0.1,iF,0.0001]",
            "trotter_slices": 1500,
            "optimization_cycles": 400,
        }
    }
    return config


def ising_to_graph(interaction_list):
    graph = nx.Graph()
    for interaction in interaction_list:
        if len(interaction) == 0:
            continue
        elif len(interaction) == 1:
            graph.add_node(*interaction)
        else:
            graph.add_edge(*interaction)
    return nx.connected_components(graph)


def test_kirchhoff(network, config):
    ising_backbone = IsingBackbone.build_ising_problem(network, config["ising_interface"])
    # only kirchhoff problem has been built
    assert ising_backbone.problem == ising_backbone._subproblems["kirchhoff"].problem
    # assert False
    num_snapshots = len(network.snapshots)
    connected_components_list = [comp for comp in ising_to_graph(ising_backbone.problem.keys())]

    # connected_components(ising_backbone.problem.keys())
    assert num_snapshots == len(connected_components_list)
    return
