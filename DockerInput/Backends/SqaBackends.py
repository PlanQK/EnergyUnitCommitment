import numpy as np
import random
from ast import literal_eval
from EnvironmentVariableManager import EnvironmentVariableManager
import siquan
from .IsingPypsaInterface import IsingPypsaInterface
from .BackendBase import BackendBase


class ClassicalBackend(BackendBase):
    def __init__(self):
        self.solver = siquan.DTSQA()
        self._metaInfo = {}

    @staticmethod
    def transformProblemForOptimizer(network):
        envMgr = EnvironmentVariableManager()
        return IsingPypsaInterface.buildCostFunction(
            network,
            float(envMgr["monetaryCostFactor"]),
            float(envMgr["kirchoffFactor"]),
            float(envMgr["minUpDownFactor"]),
        )

    @staticmethod
    def transformSolutionToNetwork(network, transformedProblem, solution):
        print(solution["state"])
        transformedProblem.addSQASolutionToNetwork(
            network, transformedProblem, solution["state"]
        )
        return network

    def optimize(self, transformedProblem):
        envMgr = EnvironmentVariableManager()
        self.solver.setSeed(int(envMgr["seed"]))
        self.solver.setTSchedule(envMgr["temperatureSchedule"])
        self.solver.setTrotterSlices(1)
        self.solver.setSteps(int(envMgr["optimizationCycles"]))
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        result["state"] = literal_eval(result["state"])
        for key in result:
            if key == "state":
                continue
            self._metaInfo[key] = result[key]
        return result

    def getMetaInfo(self):
        return self._metaInfo


class SqaBackend(ClassicalBackend):
    def optimize(self, transformedProblem):
        envMgr = EnvironmentVariableManager()
        self.solver.setSeed(int(envMgr["seed"]))
        self.solver.setHSchedule(envMgr["transverseFieldSchedule"])
        self.solver.setTSchedule(envMgr["temperatureSchedule"])
        self.solver.setTrotterSlices(int(envMgr["trotterSlices"]))
        self.solver.setSteps(int(envMgr["optimizationCycles"]))
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        result["state"] = literal_eval(result["state"])
        return result
