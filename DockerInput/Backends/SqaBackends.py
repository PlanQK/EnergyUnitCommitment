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
        self.printResults(transformedProblem, solution)
        # transformedProblem.addSQASolutionToNetwork(
        #     network, solution["state"]
        # )
        return network

    def optimize(self, transformedProblem):
        print("starting optimization...")
        self.configureSolver(HSchedule="[0]")
        tic = time.perf_counter()
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        self.metaInfo["optimizationTime"] = time.perf_counter() - tic
        # parse the entry in "state" before using it
        result["state"] = literal_eval(result["state"])
        self.writeResultsToMetaInfo(result, transformedProblem)
        print("done")
        return result

    def getMetaInfo(self):
        return self.metaInfo

    def configureSolver(self, 
            HSchedule = None,
            TSchedule = None,
            trotterSlices = None,
            steps = None
            ):
        """
        reads and sets sqa solver parameter from the environment unless
        a hyper parameter is set in the function call
        
        Args:
            HSchedule: (str) a string describing the transverse field schedule
            TSchedule: (str) a string describing the temperature schedule
            trotterSlices: (int) number of trotter slices to be used
            steps: (int) number of steps to be used
        Returns:
            (None) modifies self.solver and sets hyperparameters
        """
        try:
            self.solver.setSeed(self.metaInfo["sqaBackend"]["seed"])
        except KeyError:
            pass
        self.solver.setHSchedule(HSchedule or self.metaInfo["sqaBackend"]["transverseFieldSchedule"])
        self.solver.setTSchedule(TSchedule or self.metaInfo["sqaBackend"]["temperatureSchedule"])
        self.solver.setTrotterSlices(trotterSlices or self.metaInfo["sqaBackend"]["trotterSlices"])
        self.solver.setSteps(steps or self.metaInfo["sqaBackend"]["optimizationCycles"])
        return

    def printResults(self, transformedProblem, solution):
        """
        print the solution and then several derived values of the solution to quantify how
        good it is
        
        Args:
            transformedProblem: (IsingPypsaInterface) the isinginterface instance that encoded the spin glass
            problem
            solution: (list) list of all qubits which have spin -1 in the solution

        Returns:
            (None) Only side effect is printing information
        """
        print(f"\n--- Solution ---")
#        print(f"Qubits with spin -1: {solution['state']}")
#        print(f"Power on transmission lines: {transformedProblem.getLineValues(solution['state'])}")
        print(f"\n--- Meta parameters of the solution ---")
        print(f"Cost at each bus: {transformedProblem.individualCostContribution(solution['state'])}")
        print(f"Total Kirchhoff cost: {self.metaInfo['kirchhoffCost']}")
        print(f"Total power imbalance: {self.metaInfo['powerImbalance']}")
        print(f"Marginal Cost at each bus: {transformedProblem.individualMarginalCost(solution['state'])}")
        print(f"Total Power generated: {transformedProblem.calcTotalPowerGenerated(solution['state'])}")
        print(f"Total marginal cost: {self.metaInfo['marginalCost']}")
        print(
            f"Total cost (with constant terms): {self.metaInfo['totalCost']}\n" 
        )
        return
    
    def writeResultsToMetaInfo(self, result, transformedProblem):
        """
        This writes solution specific values of the optimizer result and the ising spin glass
        problem solution the metaInfo dictionary. Parse the value to the key "state" via 
        literal_eval before calling this function.
        
        solution:
            result: (dict) the python dictionary returned from the sqa solver
            transformedProblem: (IsingPypsaInterface) the isinginterface instance that encoded 
                    the problem into an ising sping glass problem
        Returns:
            (None) modifies self.metaInfo with solution specific parameters and values
        """
        for key in result:
            self.metaInfo[key] = result[key]
        self.metaInfo["totalCost"] = transformedProblem.calcCost(result["state"])
        self.metaInfo["solution"] = transformedProblem.calcMarginalCost(result["state"])
        self.metaInfo["kirchhoffCost"] = transformedProblem.calcKirchhoffCost(result["state"])
        self.metaInfo["powerImbalance"] = transformedProblem.calcPowerImbalance(result["state"])
        self.metaInfo["marginalCost"] = transformedProblem.calcMarginalCost(result["state"])
        self.metaInfo["sqaBackend"]["individualCost"] = transformedProblem.individualCostContribution(
                result["state"]
        )
        self.metaInfo["sqaBackend"]["eigenValues"] = sorted(transformedProblem.getHamiltonianEigenvalues()[0])
        self.metaInfo["sqaBackend"]["hamiltonian"] = transformedProblem.getHamiltonianMatrix()


class SqaBackend(ClassicalBackend):
    def optimize(self, transformedProblem):
        print("starting optimization...")
        self.configureSolver()
        tic = time.perf_counter()
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        self.metaInfo["optimizationTime"] = time.perf_counter() - tic
        # parse the entry in "state" before using it
        result["state"] = literal_eval(result["state"])
        self.writeResultsToMetaInfo(result, transformedProblem)
        print("done")
        return result
