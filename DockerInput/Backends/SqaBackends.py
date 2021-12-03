import numpy as np
import random
from ast import literal_eval
from EnvironmentVariableManager import EnvironmentVariableManager
import siquan
from .IsingPypsaInterface import IsingPypsaInterface
from .BackendBase import BackendBase


class ClassicalBackend(BackendBase):
    def __init__(self):
        envMgr = EnvironmentVariableManager()
        self.lineRepresentation=envMgr["lineRepresentation"]
        self.solver = siquan.DTSQA()
        self._metaInfo = {}
        self._metaInfo["lineRepresentation"] = self.lineRepresentation

    def validateInput(self, path, network):
        pass

    def handleOptimizationStop(self, path, network):
        pass

    def processSolution(self, network, transformedProblem, solution):
        return solution

    @staticmethod
    def transformProblemForOptimizer(network):
        print("transforming problem...")
        return IsingPypsaInterface.buildCostFunction(
            network,
        )

    @staticmethod
    def transformSolutionToNetwork(network, transformedProblem, solution):
        print(solution["state"])
        print(transformedProblem.getLineValues(solution["state"]))
        print(transformedProblem.individualCostContribution(solution["state"]))
        print(
            f"Total cost (with constant terms): {transformedProblem.calcCost(solution['state'])}"
        )
        transformedProblem.addSQASolutionToNetwork(
            network, transformedProblem, solution["state"]
        )
        return network

    def optimize(self, transformedProblem):
        print("starting optimization...")
        envMgr = EnvironmentVariableManager()
        self.solver.setSeed(int(envMgr["seed"]))
        self.solver.setTSchedule(envMgr["temperatureSchedule"])
        self.solver.setTrotterSlices(int(envMgr["trotterSlices"]))
        self.solver.setSteps(int(envMgr["optimizationCycles"]))
        self.solver.setHSchedule("[0]")
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        result["state"] = literal_eval(result["state"])
        for key in result:
            self._metaInfo[key] = result[key]

        self._metaInfo["totalCost"] = transformedProblem.calcCost(
            result["state"]
        )
        self._metaInfo[
            "individualCost"
        ] = transformedProblem.individualCostContribution(result["state"])
        print("done")
        return result

    def getMetaInfo(self):
        return self._metaInfo


class SqaBackend(ClassicalBackend):
    def optimize(self, transformedProblem):
        print("starting optimization...")
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
        for key in result:
            self._metaInfo[key] = result[key]

        self._metaInfo["totalCost"] = transformedProblem.calcCost(
            result["state"]
        )
        self._metaInfo[
            "individualCost"
        ] = transformedProblem.individualCostContribution(result["state"])
        print("done")
        return result
