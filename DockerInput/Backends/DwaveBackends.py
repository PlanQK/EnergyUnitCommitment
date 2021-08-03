import time

from dwave.cloud import Client
import dimod
import tabu
import greedy

from EnvironmentVariableManager import EnvironmentVariableManager
from .BackendBase import BackendBase
from .IsingPypsaInterface import IsingPypsaInterface


class DwaveTabuSampler(BackendBase):
    def __init__(self):
        self.solver = tabu.TabuSampler()
        self.metaInfo = {}

    def transformProblemForOptimizer(self, network):
        envMgr = EnvironmentVariableManager()
        cost = IsingPypsaInterface.buildCostFunction(
            network,
        )
        linear = {
            spins[0]: strength
            for spins, strength in cost.problem.items()
            if len(spins) == 1
        }
        # the convention is different to the sqa solver:
        # need to add a minus to the couplings
        quadratic = {
            spins: -strength
            for spins, strength in cost.problem.items()
            if len(spins) == 2
        }
        return (
            cost,
            dimod.BinaryQuadraticModel(
                linear, quadratic, 0, dimod.Vartype.SPIN
            ),
        )

    @staticmethod
    def transformSolutionToNetwork(network, transformedProblem, solution):
        # obtain the sample with the lowest energy
        bestSample = solution.first
        solutionState = [
            id for id, value in bestSample.sample.items() if value == -1
        ]
        print(solutionState)
        network = transformedProblem[0].addSQASolutionToNetwork(
            network, transformedProblem[0], solutionState
        )
        return network

    def optimize(self, transformedProblem):
        print(transformedProblem)
        print("optimize")
        tic = time.perf_counter()
        result = self.solver.sample(transformedProblem[1])
        self.metaInfo["time"] = time.perf_counter() - tic
        self.metaInfo["energy"] = result.first.energy
        return result

    def getMetaInfo(self):
        return self.metaInfo


class DwaveSteepestDescent(DwaveTabuSampler):
    def __init__(self):
        self.solver = greedy.SteepestDescentSolver()
        self.metaInfo = {}


class DwaveCloudHybrid(DwaveTabuSampler):
    def __init__(self):
        envMgr = EnvironmentVariableManager()
        self.client = Client(
            token=envMgr["dwaveAPIToken"],
        )
        self.solver = self.client.get_solver(
            "hybrid_binary_quadratic_model_version2"
        )
        self.metaInfo = {}

    def optimize(self, transformedProblem):
        print(transformedProblem)
        print("optimize")
        tic = time.perf_counter()
        asyncResponse = self.solver.sample_bqm(transformedProblem[1])
        print("Waiting for server response...")
        while True:
            if asyncResponse.done():
                break
            time.sleep(10)
        result = asyncResponse.result()
        self.metaInfo["time"] = time.perf_counter() - tic
        self.metaInfo["energy"] = result.first.energy
        return result


class DwaveCloudDirectQPU(DwaveTabuSampler):
    def __init__(self):
        envMgr = EnvironmentVariableManager()
        self.client = Client(
            token=envMgr["dwaveAPIToken"],
        )
        self.solver = self.client.get_solver("DW_2000Q_6")
        self.metaInfo = {}

    def optimize(self, transformedProblem):
        print(transformedProblem)
        print("optimize")
        tic = time.perf_counter()
        asyncResponse = self.solver.sample_bqm(transformedProblem[1])
        print("Waiting for server response...")
        while True:
            if asyncResponse.done():
                break
            time.sleep(10)
        result = asyncResponse.result()

        self.metaInfo["time"] = time.perf_counter() - tic
        self.metaInfo["energy"] = result.first.energy
        return result
