import numpy as np
import random
from ast import literal_eval
import siquan
from .IsingPypsaInterface import IsingPypsaInterface
from .BackendBase import BackendBase


class ClassicalBackend(BackendBase):
    def __init__(self, config: dict):
        super().__init__(config=config)
        self.solver = siquan.DTSQA()

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
        self.solver.setSeed(int(self.envMgr["seed"]))
        self.solver.setTSchedule(self.envMgr["temperatureSchedule"])
        self.solver.setTrotterSlices(int(self.envMgr["trotterSlices"]))
        self.solver.setSteps(int(self.envMgr["optimizationCycles"]))
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
        self._metaInfo["sqaBackend"]["individualCost"] = transformedProblem.individualCostContribution(result["state"])
        print("done")
        return result

    def getMetaInfo(self):
        return self._metaInfo


class SqaBackend(ClassicalBackend):
    def optimize(self, transformedProblem):
        print("starting optimization...")
        self.solver.setSeed(int(self.envMgr["seed"]))
        self.solver.setHSchedule(self.solver["transverseFieldSchedule"])
        self.solver.setTSchedule(self.solver["temperatureSchedule"])
        self.solver.setTrotterSlices(int(self.solver["trotterSlices"]))
        self.solver.setSteps(int(self.envMgr["optimizationCycles"]))
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        result["state"] = literal_eval(result["state"])
        for key in result:
            self._metaInfo[key] = result[key]

        self._metaInfo["totalCost"] = transformedProblem.calcCost(result["state"])
        self._metaInfo["sqaBackend"]["individualCost"] = transformedProblem.individualCostContribution(result["state"])
        print("done")
        return result
