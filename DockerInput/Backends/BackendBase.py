import abc

import pypsa

from EnvironmentVariableManager import EnvironmentVariableManager
from .Adapter import DictAdapter, YamlAdapter, JsonAdapter, StandardAdapter
from .IsingPypsaInterface import IsingBackbone


class BackendBase(abc.ABC):
    def __init__(self, adapter):
#        self.adapter = BackendBase.makeAdapter(data)
        self.setupOutputDict()
        #TODO: maybe only for DWave, QAOA and SQA? right no, no network is available at this point
#        self.isingInterface = IsingBackbone.buildIsingProblem(network=network, config=self.adapter.config)

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

    def getConfig(self) -> dict:
        return self.adapter.config

    def getResults(self) -> dict:
        return self.output

    def setupOutputDict(self):
        self.output = {"config": {"Backend": self.adapter.config["Backend"],
                                   "IsingInterface": self.adapter.config["IsingInterface"]},
                        "components": {},
                        "network": {},
                        "results": {}}

        if self.adapter.config["Backend"] in ["dwave-tabu", "dwave-greedy", "dwave-hybrid", "dwave-qpu", "dwave-read-qpu"]:
            self.output["config"]["DWaveBackend"] = self.adapter.config["DWaveBackend"]
        elif self.adapter.config["Backend"] in ["pypsa-glpk", "pypsa-fico"]:
            self.output["config"]["PypsaBackend"] = self.adapter.config["PypsaBackend"]
        elif self.adapter.config["Backend"] in ["sqa", "classical"]:
            self.output["config"]["SQABackend"] = self.adapter.config["SQABackend"]
        elif self.adapter.config["Backend"] in ["qaoa"]:
            self.output["config"]["QaoaBackend"] = self.adapter.config["QaoaBackend"]

    def buildMetaInfo(self):

        # reading variables from environment. If "int" or "float" are defined the 
        # environment variables will cast to that type. If nothing is defined, 
        # they will be kept as String-type.
        variables = {
            "fileName": "",
            "inputFile": "",
            "problemSize": "",
            "scale": "",
            "timeout": "int",  # pypsa; dwave
            "totalCost": "",  # all
            "marginalCost": "", # all
            "powerImbalance" : "",
            "kirchhoffCost": "",
            "optimizationTime" : "",
            "postprocessingTime" : "",
            "config": "",
            "dwaveBackend": {"annealing_time": "int",
                             "num_reads": "int",
                             "chain_strength": "int",
                             "programming_thermalization": "int",
                             "readout_thermalization": "int",
                             "sampleCutSize": "int",
                             "threshold": "float",
                             "strategy": "",
                             "postprocess": "",
                             "annealReadRatio": "",  # output
                             "totalAnnealTime": "",
                             "mangledTotalAnnealTime": "",
                             "LowestEnergy": "",
                             "LowestFlow": "",
                             "ClosestFlow": "",
                             "cutSamplesCost": "",
                             "optimizedStrategySample": "",
                             "solver_id": "",
                             "energy": ""
                             },
            "isingInterface": {"kirchhoffFactor": "float",
                               "monetaryCostFactor": "float",
                               "minUpDownFactor": "float",
                               "problemFormulation": "",
                               "offsetEstimationFactor": "",
                               "estimatedCostFactor": "",
                               "offsetBuildFactor": "",
                               "hamiltonian": "",
                               "eigenValues": "",
                               },
            "pypsaBackend": {"solver_name": "",
                             "terminationCondition" : "",
                             },
            "sqaBackend": {"individualCost": "",
                           "seed": "int",
                           "transverseFieldSchedule": "",
                           "temperatureSchedule": "",
                           "trotterSlices": "int",
                           "optimizationCycles": "int",
                           },
            "qaoaBackend": ""
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
                    self.metaInfo[var][dictVar] = populateMetaInfo(
                            varType=variables[var][dictVar], varName=dictVar
                    )
            else:
                self.metaInfo[var] = populateMetaInfo(varType=variables[var], varName=var)

        self.metaInfo["fileName"] = self.envMgr["outputInfo"]
        self.metaInfo["inputFile"] = "_".join(self.metaInfo["fileName"].split("_")[1:5])
        self.metaInfo["problemSize"] = int(self.metaInfo["fileName"].split("_")[2])
        self.metaInfo["scale"] = int(self.metaInfo["fileName"].split("_")[4][:-3])

        self.metaInfo["config"] = {}
        for key in self.config:
            if key != "APItoken":
                self.metaInfo["config"][key] = self.config[key]

        return
