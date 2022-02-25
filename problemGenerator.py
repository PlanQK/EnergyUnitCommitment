"""This script creates random instances of pypsa networks that need to be optimized.
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

CONFIG = {
    # the variance indicates how different the random variables are to each other
    # for variance=0.2 at most a random variable is 20% off
    "variance": 0.4,
    # the number of buses that a typical problem has
    "numBuses": 2,
    # conection number gets divided by number of buses
    "connectionNumber": 1.0,
    # the average number of generators there are for each bus
    "generatorsPerBus": 1,
    # average Energy required by loads for one bus:
    "averageRequiredEnergy": 10,
    # average Energy generated by all Generators on one bus:
    "averageProducedEnergy": 30,
    # average generator cost per produced energy (this can be float)
    "avMarginalCost": 5.0,
    # number of different time slices
    "snapshots": 1,
    # scaling factor for line capacity in percent
    "lineScale": 20,
}


def randomFactor():
    """small helper function to generate a random factor taking
    into account the variance from the CONFIG.
    """
    return 1 + 2 * CONFIG["variance"] * (random.random() - 0.5)


class Partitioner:
    def __init__(self, totalValue, numPartitions):
        """Find numPartitions numbers such that
        the sum of each number is totalValue.

        The maximum deviation allowed is given by variance.
        """
        self.totalValue = int(totalValue)
        self.numPartitions = int(numPartitions)
        self.average = int(self.totalValue / self.numPartitions)
        self.maxDeviation = round(self.average * CONFIG["variance"])
        if self.numPartitions % 2:
            self.upperHalf = [self.average] * int(self.numPartitions / 2)
            self.lowerHalf = [self.average] * int(self.numPartitions / 2)
            self.middle = [
                self.average,
            ]
        else:
            self.upperHalf = [self.average] * int(self.numPartitions / 2)
            self.lowerHalf = [self.average] * int(self.numPartitions / 2)
            self.middle = []
        missing = self.totalValue - (
            sum(self.upperHalf) + sum(self.lowerHalf) + sum(self.middle)
        )
        # TODO: find a better way to distribute the rest
        for i in range(missing):
            pos = random.randint(0, len(self.lowerHalf) - 1)
            self.lowerHalf[pos] += 1
        self.induceRandomness()

    def induceRandomness(self):
        for index in range(len(self.lowerHalf)):
            value = random.randint(0, self.maxDeviation)
            self.lowerHalf[index] -= value
            self.upperHalf[index] += value

    def returnPartition(self):
        partition = self.lowerHalf + self.middle + self.upperHalf
        random.shuffle(partition)
        return partition


class ProblemInstance:
    def __init__(self) -> None:
        self.network = pypsa.Network()
        self._createBusesAndLoads()
        self._createGenerators()
        self._createLines()

    def _createBusesAndLoads(self) -> None:
        loads = []
        for _ in range(CONFIG["snapshots"]):
            numBuses =  CONFIG["numBuses"]
#            numBuses = randomFactor() * CONFIG["numBuses"]
            loads.append(
                Partitioner(
                    CONFIG["averageRequiredEnergy"] * numBuses,
                    numBuses,
                ).returnPartition()
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

    def _createGenerators(self) -> None:
        generatorsPerBus = [
            int(randomFactor() * CONFIG["generatorsPerBus"])
            for i in range(len(self.network.buses))
        ]
        # time varying
        generatorOutput = []
        for _ in range(CONFIG["snapshots"]):
            generatorOutput.append(
                Partitioner(
                    CONFIG["averageProducedEnergy"] * len(self.network.buses),
                    sum(generatorsPerBus),
                ).returnPartition()
            )
        genOutputIndex = 0
        for busId in range(len(generatorsPerBus)):
            for generatorId in range(generatorsPerBus[busId]):
                self.network.add(
                    "Generator",
                    f"Gen_{busId}_{generatorId}",
                    bus=f"Bus_{busId}",
                    p_max_pu=[
                        element[genOutputIndex] for element in generatorOutput
                    ],
                    p_nom=1,
                    marginal_cost=randomFactor() * CONFIG["avMarginalCost"],
                )
                genOutputIndex += 1

    def _createLines(self):
        """Each bus needs to be connected with at least 1 line."""
        averageLoad = CONFIG["averageRequiredEnergy"]
        lineScale = float(CONFIG["lineScale"])/100.0
        buses = [i for i in range(len(self.network.buses))]
        for busId1 in range(len(self.network.buses)):
            tmpBuses = buses.copy()
            tmpBuses.remove(busId1)
            busId2 = random.choice(tmpBuses)
            self.network.add(
                "Line",
                f"Line_{busId1}_{busId2}",
                bus0=f"Bus_{busId1}",
                bus1=f"Bus_{busId2}",
                s_nom=int(randomFactor() * lineScale * averageLoad),
            )
        # add some additional lines as well
        for busId1 in range(len(self.network.buses)):
            for busId2 in range(len(self.network.buses)):
                if busId1 == busId2:
                    continue
                if len(
                    self.network.lines[
                        (
                            (self.network.lines.bus0 == f"Bus_{busId2}")
                            & (self.network.lines.bus1 == f"Bus_{busId1}")
                        )
                        | (
                            (self.network.lines.bus0 == f"Bus_{busId1}")
                            & (self.network.lines.bus1 == f"Bus_{busId2}")
                        )
                    ]
                ):
                    continue
                if random.random() < CONFIG["connectionNumber"] / len(
                    self.network.buses
                ):
                    self.network.add(
                        "Line",
                        f"Line_{busId1}_{busId2}",
                        bus0=f"Bus_{busId1}",
                        bus1=f"Bus_{busId2}",
                        s_nom=int(randomFactor() * lineScale * averageLoad),
                    )

    def exportNetwork(self, fileName: str) -> None:
        self.network.export_to_netcdf(fileName)


usageString = """
usage: problemGenerator.py num_problems
       problemGenerator.py num_problems scale
"""


def main():
    if len(sys.argv) == 1 or len(sys.argv) > 3: 
        print(usageString)
        exit(1)
    global CONFIG
    if len(sys.argv) == 3:
        CONFIG["lineScale"] = int(sys.argv[2])
    for numBuses in range(2,25,1):
        CONFIG["numBuses"] = numBuses
        for i in range(int(sys.argv[1])):
            if ospath.isfile(f"TestProblems/220124cost5input_{CONFIG['numBuses']}_{i}_{CONFIG['lineScale']}.nc"):
                print(f"skipping: {CONFIG['numBuses']}_{i}_{CONFIG['lineScale']}")
                continue
            newProblem = ProblemInstance()
            newProblem.exportNetwork(
                f"TestProblems/220124cost5input_{CONFIG['numBuses']}_{i}_{CONFIG['lineScale']}.nc"
            )
    return


if __name__ == "__main__":
    main()
