"""
This module contains a test concerning the functionality of the QuboTransformer class.
"""
import typing
import os

import pytest
import pypsa

import networkx as nx

from src.libs.qubo_transformator import QuboTransformator


@pytest.fixture(scope="session")
def network_path():
    """
    Find the path of the networks folder independent of the working
    directory of the pytest call
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + "/../input/networks/defaultnetwork.nc"
    return path


@pytest.fixture(scope="session")
def network(network_path):
    """
    Load the default network in the repo and cut any snapshot
    after the first three
    """
    default = pypsa.Network(network_path)
    default.snapshots = default.snapshots[:3]
    return default


@pytest.fixture
def config():
    "Set configuration parameters"
    cfg = {
        "backend": "sqa",
        "ising_interface": {},
        "snapshots": 3,
        "sqa_backend": {
            "transverse_field_schedule": "[8.0,0.0]",
            "temperature_schedule": "[0.1,iF,0.0001]",
            "trotter_slices": 1000,
            "optimization_cycles": 400,
        },
    }
    return cfg


def ising_to_graph(interaction_list):
    """
    Build the connected components of the ising interactions
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
    Test that without extra problems added, only the kirchhoff problem
    is built and the interactions are distinct across snapshots
    """
    ising_backbone = QuboTransformator.transform_network_to_qubo(
        network, config["ising_interface"]
    )
    # only kirchhoff problem has been built
    assert (
        ising_backbone.get_ising_coefficients()
        == ising_backbone.get_subproblems()["kirchhoff"]._ising_coefficients
    )

    # kirchhoff interactions are not connected across different snapshots
    num_snapshots = len(network.snapshots)
    connected_components_list = [
        comp for comp in ising_to_graph(ising_backbone.get_ising_coefficients().keys())
    ]
    assert num_snapshots == len(connected_components_list)
    return
