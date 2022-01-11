import abc

from EnvironmentVariableManager import EnvironmentVariableManager


class BackendBase(abc.ABC):
    def __init__(self):
        self.envMgr = EnvironmentVariableManager()
        self.metaInfo = {}
        self.buildMetaInfo()

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

    def buildMetaInfo(self):

        # reading variables from environment??
        variables = {
            "fileName": "",
            "problemSize": "",
            "scale": "",
            "annealing_time": "int",
            "num_reads": "int",
            "timeout": "int",
            "chain_strength": "int",
            "programming_thermalization": "int",
            "readout_thermalization": "int",
            "lineRepresentation": "int",
            "maxOrder": "int",
            "sampleCutSize": "int",
            "sampleValue": "",
            "minChoice": "",
            "kirchhoffFactor": "float",
            "slackVarFactor": "float",
            "monetaryCostFactor": "float",
            "threshold": "float",
            "minUpDownFactor": "float",
            "time": "",
            "energy": "",
            "strategy": "string",
            "postprocess": "string",
            "totalCost": "",
            "individualCost": "",
            "annealReadRatio": "",
            "totalAnnealTime": "",
            "mangledTotalAnnealTime": "",
            "LowestEenrgy": "",
            "LowestFlow": "",
            "ClosestFlow": "",
            "cutSamplesCost": "",
            "optimizedStrategySample": "",
            "solver_id": ""
        }

        for var in variables:
            if variables[var] == "int":
                setattr(self, var, int(self.envMgr[var]))
                self.metaInfo[var] = int(self.envMgr[var])
            elif variables[var] == "float":
                setattr(self, var, float(self.envMgr[var]))
                self.metaInfo[var] = float(self.envMgr[var])
            else:
                setattr(self, var, self.envMgr[var])
                self.metaInfo[var] = self.envMgr[var]

        return