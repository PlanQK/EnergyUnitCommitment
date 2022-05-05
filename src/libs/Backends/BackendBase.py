import abc

import pypsa

from .InputReader import InputReader
from datetime import datetime


class BackendBase(abc.ABC):
    def __init__(self, reader: InputReader):
        self.reader = reader
        self.network = reader.getNetwork()
        self.networkName = reader.getNetworkName()
        self.config = reader.getConfig()
        self.setupOutputDict()
        self.transformedProblem = None

    @abc.abstractmethod
    def transformProblemForOptimizer(self):  # -> set self.transformedProblem
        pass

    @abc.abstractstaticmethod
    def transformSolutionToNetwork( solution) -> pypsa.Network:
        pass

    def processSolution(self, solution) -> dict:
        self.output["results"]["postprocessingTime"] = 0.0
        return solution

    @abc.abstractmethod
    def optimize(self):
        pass

    def validateInput(self, path):
        pass

    # TODO: implemented in DWave, but not used right now. (Can we have a blacklist on PlanQK?)
    def handleOptimizationStop(self, path):
        pass

    def getConfig(self) -> dict:
        """
        Getter function for the config-dictionary.

        Returns:
            (dict) The config used for the current problem.
        """
        return self.config

    def getTime(self) -> str:
        """
        Getter function for the current time.

        Returns:
            (str) The current time in the format YYYY-MM-DD_hh-mm-ss
        """
        now = datetime.today()
        return f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"

    def getOutput(self) -> dict:
        """
        Getter function for the output-dictionary.

        Returns:
            (dict) The output (result) of the current problem.
        """
        self.output["end_time"] = self.getTime()
        return self.output

    def printSolverspecificReport(self):
        pass

    def printReport(self) -> None:
        """
        Prints a short report with general information about the solution.
        Returns:
            None.
        """
        print(f"\n--- General information of the solution ---")
        print(
            f'Kirchhoff cost at each bus: {self.output["results"].get("individualKirchhoffCost","N/A")}'
        )
        print(
            f'Total Kirchhoff cost: {self.output["results"].get("kirchhoffCost","N/A")}'
        )
        print(
            f'Total power imbalance: {self.output["results"].get("powerImbalance","N/A")}'
        )
        print(
            f'Total Power generated: {self.output["results"].get("totalPower","N/A")}'
        )
        print(
            f'Total marginal cost: {self.output["results"].get("marginalCost","N/A")}'
        )
        self.printSolverspecificReport()
        print("---")

    def setupOutputDict(self) -> None:
        """
        Creates an 'output' attribute in self in which to save results and configuration
        data. The config entry is another dictionary with 3 keys: 'Backend' has config
        data that all backends share, 'IsingInterface' has config data of the class
        used to convert a unit commitment problem into an ising spin problem
        and a key named in `BackendToSolver` for backend specific configurations

        Returns:
            None. (over)writes the attribute `output` with a dictionary containing
            configuration data and empty fields to insert results into later on.
        """
        startTime = self.getTime()
        for backend, solverList in self.reader.BackendToSolver.items():
            if self.config["Backend"] in solverList:
                self.output = {
                    "start_time": startTime,
                    "end_time": None,
                    "file_name": "_".join(
                        [self.networkName, self.config["Backend"], startTime + ".json"]
                    ),
                    "config": {
                        "Backend": self.config["Backend"],
                        "BackendType": backend,
                        "BackendConfig": self.config[backend],
                        "IsingInterface": self.config["IsingInterface"],
                    },
                    "components": {},
                    "network": {},
                    "results": {},
                }
                return
