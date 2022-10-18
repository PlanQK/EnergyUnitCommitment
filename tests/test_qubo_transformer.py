import pytest
import typing

import pypsa

import networkx as nx

from src.libs.qubo_transformator import QuboTransformator

import os


@pytest.fixture(scope='session')
def network_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    network_path = dir_path + "/../networks/defaultnetwork.nc"
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
        "ising_interface": {},
        "snapshots": 3,
        "sqa_backend": {
            "transverse_field_schedule": "[8.0,0.0]",
            "temperature_schedule": "[0.1,iF,0.0001]",
            "trotter_slices": 1000,
            "optimization_cycles": 400,
        }
    }
    return config


def ising_to_graph(interaction_list):
    """
    build the connected components of the ising interactions
    """
    graph = nx.Graph()
    for interaction in interaction_list:
        if len(interaction) == 0:
            continue
        elif len(interaction) == 1:
            graph.add_node(*interaction)
        else:
            graph.add_edge(*interaction)
    return nx.connected_components(graph)


def test_kirchhoff_distinct_snapshots(network, config):
    """
    test that without extra problems added, only the kirchhoff problem
    is build and the interactions are distinct across snapshots
    """
    qubo_transformator = QuboTransformator(
        network=network,
        config=config["ising_interface"]
    )
    ising_backbone = qubo_transformator.transform_network_to_qubo()
    # only kirchhoff problem has been built
    assert ising_backbone.ising_coefficients == ising_backbone._subproblems["kirchhoff"].ising_coefficients

    # kirchhoff interactions are not connected across different snapshots
    num_snapshots = len(network.snapshots)
    connected_components_list = [comp for comp in ising_to_graph(ising_backbone.ising_coefficients.keys())]
    assert num_snapshots == len(connected_components_list)
    return
