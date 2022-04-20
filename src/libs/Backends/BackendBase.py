import abc

from .InputReader import InputReader
from datetime import datetime
from .IsingPypsaInterface import IsingBackbone


class BackendBase(abc.ABC):
    def __init__(self, reader: InputReader):
        self.network = reader.getNetwork()
        self.networkName = reader.getNetworkName()
        self.config = reader.getConfig()
        self.setupOutputDict()

    @abc.abstractmethod
    def transformProblemForOptimizer(self, network):
        pass

    @abc.abstractstaticmethod
    def transformSolutionToNetwork(network, transformedProblem, solution):
        pass

    @abc.abstractmethod
    def processSolution(self, network, transformedProblem, solution):
        pass

    @abc.abstractmethod
    def optimize(self, transformedProblem):
        pass

    @abc.abstractmethod
    def getOutput(self):
        pass

    @abc.abstractmethod
    def validateInput(self, path, network):
        pass

    @abc.abstractmethod
    def handleOptimizationStop(self, path, network):
        pass

    def getConfig(self) -> dict:
        return self.config

    def getResults(self) -> dict:
        return self.output

    def setupOutputDict(self):
        self.output = {"start_time": None,
                       "end_time": None,
                       "file_name": None,
                       "config": {"Backend": self.config["Backend"],
                                  "IsingInterface": self.config["IsingInterface"]},
                       "components": {},
                       "network": {},
                       "results": {}}

        if self.config["Backend"] in ["dwave-tabu", "dwave-greedy", "dwave-hybrid", "dwave-qpu", "dwave-read-qpu"]:
            self.output["config"]["DWaveBackend"] = self.config["DWaveBackend"]
        elif self.config["Backend"] in ["pypsa-glpk", "pypsa-fico"]:
            self.output["config"]["PypsaBackend"] = self.config["PypsaBackend"]
        elif self.config["Backend"] in ["sqa", "classical"]:
            self.output["config"]["SqaBackend"] = self.config["SqaBackend"]
        elif self.config["Backend"] in ["qaoa"]:
            self.output["config"]["QaoaBackend"] = self.config["QaoaBackend"]

        self.output["start_time"] = self.getTime()
        self.output["file_name"] = self.networkName + "_" + self.output["start_time"]

    def getTime(self) -> str:
        now = datetime.today()
        dateTimeStr = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"

        return dateTimeStr
