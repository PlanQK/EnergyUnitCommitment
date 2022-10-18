import pypsa

def create_network(load_list):
    network = pypsa.Network(snapshots=range(len(load_list)))
    network.add("Bus","bus")
    network.add(
        "Generator",
        "gen_1",
        bus="bus",
        committable=True,
        p_min_pu=1,
        marginal_cost=15,
        p_nom=4,
    )
    network.add(
        "Generator",
        "gen_2",
        bus="bus",
        committable=True,
        p_min_pu=1,
        marginal_cost=10,
        p_nom=3,
    )
    network.add(
        "Load",
        "load",
        bus="bus",
        p_set=load_list
    )
    return network

