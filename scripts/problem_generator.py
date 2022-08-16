"""this script creates random instances of pypsa networks that need to be optimized.
The networks consist of buses, generators, lines and loads.

Configuration parameters are found in the CONFIG dictionary of this script. A short
explanation for each parameter is also given. These parameters influence if the
generated problems have a solution or not.

For simplification no floating point values are used.

I recommend thinking very carefully before changing the parameters.
"""
import sys
import pypsa
import random
from os import path as ospath

from datetime import datetime

CONFIG = {
    # the variance indicates how different the random variables are to each other
    # for variance=0.2 at most a random variable is 20% off
    "variance": 0.4,
    # the number of buses that a typical problem has
    "num_buses": 2,
    # connection number gets divided by number of buses
    "connection_number": 1.2,
    # the average number of generators there are for each bus
    "generators_per_bus": 4,
    # average Energy required by loads for one bus:
    "average_required_energy": 10,
    # average Energy generated by all Generators on one bus:
    "average_produced_energy": 30,
    # average generator cost per produced energy (this is a float)
    "av_marginal_cost": 5.0,
    # number of different time slices
    "snapshots": 3,
    # scaling factor for line capacity in percent
    "line_scale": 20,
    # location where to save the network starting at git root
    "savepath": "networks",
    # prefix of the filename. If this is empty the prefix will be set to
    # {date}_network , with the date win yyyymmdd format
    "prefix": "",
}


def random_factor():
    """small helper function to generate a random factor taking
    into account the variance from the CONFIG.
    """
    return 1 + 2 * CONFIG["variance"] * (random.random() - 0.5)


def normalize_output(output_list):
    """
    normalize the output_list by scaling them such that the biggest input is 1.
    Then return the used scalar and the resulting list. This brings it in line with
    the pypsa standard which specifies generator output over multiple snapshots
    by specifying the maximum power a generator can produce. Variation on the
    available power generation at any snapshot is then given by a percentage
    of the maximum power

    Args:
        output_list: (list[float]) the list to be scaled
    Returns:
        (float, list[float]) The maximum entry of the list and the list divided by that
    """
    max_element = max(*output_list)
    print(f"max::{max_element}") 
    print(f"output_list::{output_list}")
    return max_element, [value / float(max_element) for value in output_list]


class Partitioner:
    def __init__(self, total_value, num_partitions):
        """Find num_partitions numbers such that
        the sum of each number is total_value.

        The maximum deviation allowed is given by variance.
        """
        self.total_value = int(total_value)
        self.num_partitions = int(num_partitions)
        self.average = int(self.total_value / self.num_partitions)
        self.max_deviation = round(self.average * CONFIG["variance"])
        if self.num_partitions % 2:
            self.upper_half = [self.average] * int(self.num_partitions / 2)
            self.lower_half = [self.average] * int(self.num_partitions / 2)
            self.middle = [
                self.average,
            ]
        else:
            self.upper_half = [self.average] * int(self.num_partitions / 2)
            self.lower_half = [self.average] * int(self.num_partitions / 2)
            self.middle = []
        missing = self.total_value - (
                sum(self.upper_half) + sum(self.lower_half) + sum(self.middle)
        )
        # TODO: find a better way to distribute the rest
        for _ in range(missing):
            pos = random.randint(0, len(self.lower_half) - 1)
            self.lower_half[pos] += 1
        self.induce_randomness()

    def induce_randomness(self):
        for index in range(len(self.lower_half)):
            value = random.randint(0, self.max_deviation)
            self.lower_half[index] -= value
            self.upper_half[index] += value

    def return_partition(self):
        partition = self.lower_half + self.middle + self.upper_half
        random.shuffle(partition)
        return partition


class ProblemInstance:
    def __init__(self) -> None:
        self.network = pypsa.Network()
        self.network.set_snapshots(range(CONFIG["snapshots"]))
        self._create_buses_and_loads()
        self._create_generators()
        self._create_lines()

    def _create_buses_and_loads(self) -> None:
        loads = []
        for _ in range(CONFIG["snapshots"]):
            num_buses = CONFIG["num_buses"]
            #            num_buses = random_factor() * CONFIG["num_buses"]
            loads.append(
                Partitioner(
                    CONFIG["average_required_energy"] * num_buses,
                    num_buses,
                ).return_partition()
            )
        print(loads)
        for i in range(len(loads[0])):
            self.network.add(
                "Bus",
                f"Bus_{i}",
            )
            self.network.add(
                "Load",
                f"Load_{i}",
                bus=f"Bus_{i}",
                p_set=[element[i] for element in loads],
                q_set=[element[i] for element in loads],
            )

    def _create_generators(self) -> None:
        generators_per_bus = [
            int(random_factor() * CONFIG["generators_per_bus"])
            for _ in range(len(self.network.buses))
        ]
        # time varying
        generator_output = []
        for _ in range(CONFIG["snapshots"]):
            generator_output.append(
                Partitioner(
                    CONFIG["average_produced_energy"] * len(self.network.buses),
                    sum(generators_per_bus),
                ).return_partition()
            )
        gen_output_index = 0
        for bus_id in range(len(generators_per_bus)):
            for generator_id in range(generators_per_bus[bus_id]):
                p_nom, p_max_pu = normalize_output(
                    [element[gen_output_index] for element in generator_output]
                )
                self.network.add(
                    "Generator",
                    f"Gen_{bus_id}_{generator_id}",
                    bus=f"Bus_{bus_id}",
                    p_max_pu=p_max_pu,
                    p_nom=p_nom,
                    marginal_cost=random_factor() * CONFIG["av_marginal_cost"],
                )
                gen_output_index += 1

    def _create_lines(self):
        """Each bus needs to be connected with at least 1 line."""
        average_load = CONFIG["average_required_energy"]
        line_scale = float(CONFIG["line_scale"]) / 100.0
        buses = [i for i in range(len(self.network.buses))]
        for bus_id_1 in range(len(self.network.buses)):
            tmp_buses = buses.copy()
            tmp_buses.remove(bus_id_1)
            bus_id_2 = random.choice(tmp_buses)
            self.network.add(
                "Line",
                f"Line_{bus_id_1}_{bus_id_2}",
                bus0=f"Bus_{bus_id_1}",
                bus1=f"Bus_{bus_id_2}",
                s_nom=int(random_factor() * line_scale * average_load),
            )
        # add some additional lines as well
        for bus_id_1 in range(len(self.network.buses)):
            for bus_id_2 in range(len(self.network.buses)):
                if bus_id_1 == bus_id_2:
                    continue
                if len(
                        self.network.lines[
                            (
                                    (self.network.lines.bus0 == f"Bus_{bus_id_2}")
                                    & (self.network.lines.bus1 == f"Bus_{bus_id_1}")
                            )
                            | (
                                    (self.network.lines.bus0 == f"Bus_{bus_id_1}")
                                    & (self.network.lines.bus1 == f"Bus_{bus_id_2}")
                            )
                        ]
                ):
                    continue
                if random.random() < CONFIG["connection_number"] / len(
                        self.network.buses
                ):
                    self.network.add(
                        "Line",
                        f"Line_{bus_id_1}_{bus_id_2}",
                        bus0=f"Bus_{bus_id_1}",
                        bus1=f"Bus_{bus_id_2}",
                        s_nom=int(random_factor() * line_scale * average_load),
                    )

    def export_network(self, file_name: str) -> None:
        self.network.export_to_netcdf(file_name)


usage_string = """
usage: problemGenerator.py num_problems
       problemGenerator.py num_problems scale
"""


def main():
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        print(usage_string)
        exit(1)
    global CONFIG
    if len(sys.argv) == 3:
        CONFIG["line_scale"] = int(sys.argv[2])
    now = datetime.today()
    prefix = CONFIG.get('prefix', "")
    if CONFIG["prefix"] == "":
        prefix = f"{now.year}{now.month:02d}{now.day:02d}_network"
    for num_buses in range(5, 6, 1):
        CONFIG["num_buses"] = num_buses
        for i in range(int(sys.argv[1])):
            current_networkname = f"{CONFIG['savepath']}/{prefix}_{CONFIG['num_buses']}_{i}_{CONFIG['line_scale']}.nc"
            if ospath.isfile(current_networkname):
                print(f"skipping {prefix} with settings: {CONFIG['num_buses']}_{i}_{CONFIG['line_scale']}")
                continue
            new_problem = ProblemInstance()
            new_problem.export_network(
               current_networkname
            )
    return


if __name__ == "__main__":
    main()
