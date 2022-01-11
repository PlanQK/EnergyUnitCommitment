import abc

from EnvironmentVariableManager import EnvironmentVariableManager


class BackendBase(abc.ABC):
    def __init__(self, network):
        envMgr = EnvironmentVariableManager()

    @abc.abstractstaticmethod
    def transformProblemForOptimizer(network):
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
    def getMetaInfo(self):
        pass

    @abc.abstractmethod
    def validateInput(self, path, network):
        pass

    @abc.abstractmethod
    def handleOptimizationStop(self, path, network):
        pass

    def buildMetaInfo(self, metaInfo: dict):
        intVars = [
            "annealing_time",
            "num_reads",
            "timeout",
            "chain_strength",
            "programming_thermalization",
            "readout_thermalization",
            "lineRepresentation",
            "maxOrder",
            "num_reads",
            "sampleCutSize",
        ]
        for var in intVars:
            setattr(self, var, int(self.envMgr[var]))
            self.metaInfo[var] = int(self.envMgr[var])

        floatVars = [
            "kirchhoffFactor",
            "slackVarFactor",
            "monetaryCostFactor",
            "threshold",
        ]
        for var in floatVars:
            setattr(self, var, float(self.envMgr[var]))
            self.metaInfo[var] = float(self.envMgr[var])

        stringVars = [
            "strategy",
            "postprocess",
        ]
        for var in stringVars:
            setattr(self, var, self.envMgr[var])
            self.metaInfo[var] = self.envMgr[var]
