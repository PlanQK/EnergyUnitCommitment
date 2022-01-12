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

        # reading variables from environment. If "int" or "float" are defined the environment variables will cast to
        # that type. If nothing is defined, they will be kept as String-type.
        variables = {
            "fileName": "",
            "inputFile": "",
            "problemSize": "",
            "scale": "",
            "dwaveBackend": {"annealing_time": "int",
                             "num_reads": "int",
                             "chain_strength": "int",
                             "programming_thermalization": "int",
                             "readout_thermalization": "int",
                             "lineRepresentation": "int",
                             "maxOrder": "int",
                             "sampleCutSize": "int",
                             },
            "timeout": "int",#pypsa; dwave
            "kirchhoffFactor": "float",#ising
            "slackVarFactor": "float",#ising
            "monetaryCostFactor": "float",#ising
            "threshold": "float",#dwave
            "minUpDownFactor": "float",#ising
            "strategy": "",#dwave
            "postprocess": "",#dwave
            "pypsaBackend": {"solver_name": "",
                             "slack_gen_penalty": ""},
            "sampleValue": "",
            "minChoice": "",
            "time": "",#pypsa; dwave
            "energy": "",
            "totalCost": "",
            "individualCost": "",
            "annealReadRatio": "",
            "totalAnnealTime": "",
            "mangledTotalAnnealTime": "",
            "LowestEnergy": "",
            "LowestFlow": "",
            "ClosestFlow": "",
            "cutSamplesCost": "",
            "optimizedStrategySample": "",
            "solver_id": ""
        }

        def populateMetaInfo(varType: str, varName: str):
            if varType == "int":
                return int(self.envMgr[varName])
            elif varType == "float":
                return float(self.envMgr[varName])
            else:
                return self.envMgr[varName]

        for var in variables:
            if isinstance(variables[var], dict):
                self.metaInfo[var] = {}
                for dictVar in variables[var]:
                    self.metaInfo[var][dictVar] = populateMetaInfo(varType=variables[var][dictVar], varName=dictVar)
            else:
                self.metaInfo[var] = populateMetaInfo(varType=variables[var], varName=var)


        self.metaInfo["fileName"] = self.envMgr["outputInfo"]
        self.metaInfo["inputFile"] = "_".join(self.metaInfo["fileName"].split("_")[1:5])
        self.metaInfo["problemSize"] = int(self.metaInfo["fileName"].split("_")[2])
        self.metaInfo["scale"] = int(self.metaInfo["fileName"].split("_")[4][:-3])

        return

        if variables[var] == "int":
            setattr(self, var, int(self.envMgr[var]))
            self.metaInfo[var] = int(self.envMgr[var])
        elif variables[var] == "float":
            setattr(self, var, float(self.envMgr[var]))
            self.metaInfo[var] = float(self.envMgr[var])
        else:
            setattr(self, var, self.envMgr[var])
            self.metaInfo[var] = self.envMgr[var]

