import numpy as np
import random
from ast import literal_eval
import siquan
from .IsingPypsaInterface import IsingPypsaInterface
from .BackendBase import BackendBase
import time


class ClassicalBackend(BackendBase):
    def __init__(self, config: dict):
        super().__init__(config=config)
        self.solver = siquan.DTSQA()

    def validateInput(self, path, network):
        pass

    def handleOptimizationStop(self, path, network):
        pass

    def processSolution(self, network, transformedProblem, solution):
        self.metaInfo["postprocessingTime"] = 0.0
        return solution

    @staticmethod
    def transformProblemForOptimizer(network):
        print("transforming problem...")
        return IsingPypsaInterface.buildCostFunction(
            network,
        )

    def transformSolutionToNetwork(self, network, transformedProblem, solution):
        print(solution["state"])
        print(transformedProblem.getLineValues(solution["state"]))
        print(transformedProblem.individualCostContribution(solution["state"]))
        kirchhoffCost = 0.0
        for key, val in transformedProblem.individualCostContribution(solution["state"],silent=False).items():
            kirchhoffCost += val 
        print(f"Total Kirchhoff cost: {kirchhoffCost}")
        print(transformedProblem.individualMarginalCost(solution["state"]))
        marginalCost = transformedProblem.calcMarginalCost(solution["state"])
        print(f"TOTAL MARGINAL COST: {marginalCost}")
        self.metaInfo["marginalCost"] = transformedProblem.calcMarginalCost(solution["state"])


        print(
            f"Total cost (with constant terms): {transformedProblem.calcCost(solution['state'])}"
        )
        # transformedProblem.addSQASolutionToNetwork(
        #     network, solution["state"]
        # )
        return network

    def optimize(self, transformedProblem):
        print("starting optimization...")
        try:
            self.solver.setSeed(self.metaInfo["sqaBackend"]["seed"])
        except KeyError:
            pass
        self.solver.setTSchedule(self.metaInfo["sqaBackend"]["temperatureSchedule"])
        self.solver.setTrotterSlices(self.metaInfo["sqaBackend"]["trotterSlices"])
        self.solver.setSteps(self.metaInfo["sqaBackend"]["optimizationCycles"])
        self.solver.setHSchedule("[0]")
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        result["state"] = literal_eval(result["state"])
        for key in result:
            self.metaInfo[key] = result[key]

        self.metaInfo["totalCost"] = transformedProblem.calcCost(
            result["state"]
        )
        self.metaInfo["marginalCost"] = transformedProblem.calcMarginalCost(result["state"])
        self.metaInfo["sqaBackend"]["individualCost"] = transformedProblem.individualCostContribution(result["state"])
        print("done")
        return result

    def getMetaInfo(self):
        return self.metaInfo


class SqaBackend(ClassicalBackend):
    def optimize(self, transformedProblem):
        print("starting optimization...")
        try:
            self.solver.setSeed(self.metaInfo["sqaBackend"]["seed"])
        except KeyError:
            pass
        self.solver.setHSchedule(self.metaInfo["sqaBackend"]["transverseFieldSchedule"])
        self.solver.setTSchedule(self.metaInfo["sqaBackend"]["temperatureSchedule"])
        self.solver.setTrotterSlices(self.metaInfo["sqaBackend"]["trotterSlices"])
        self.solver.setSteps(self.metaInfo["sqaBackend"]["optimizationCycles"])
        tic = time.perf_counter()
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        self.metaInfo["optimizationTime"] = time.perf_counter() - tic
        result["state"] = literal_eval(result["state"])
        for key in result:
            self.metaInfo[key] = result[key]

        self.metaInfo["totalCost"] = transformedProblem.calcCost(result["state"])
        self.metaInfo["marginalCost"] = transformedProblem.calcMarginalCost(result["state"])
        self.metaInfo["sqaBackend"]["individualCost"] = transformedProblem.individualCostContribution(result["state"])
        print("done")
        return result
