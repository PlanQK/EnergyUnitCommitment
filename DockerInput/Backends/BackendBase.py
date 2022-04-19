import abc

from .InputReader import InputReader
from .IsingPypsaInterface import IsingBackbone


class BackendBase(abc.ABC):
    def __init__(self, inputReader: InputReader):
        self.network = inputReader.getNetwork()
        self.config = inputReader.getConfig()
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
        self.output = {"config": {"Backend": self.config["Backend"],
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
