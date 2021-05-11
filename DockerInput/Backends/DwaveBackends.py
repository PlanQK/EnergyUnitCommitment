import time

from dwave.cloud import Client
import dimod
import tabu
import greedy

from EnvironmentVariableManager import EnvironmentVariableManager
from .BackendBase import BackendBase
from .IsingPypsaInterface import IsingPypsaInterface


class DwaveClassicalBackend(BackendBase):
    def __init__(self):
        self.solver = tabu.TabuSampler()
        self.metaInfo = {}

    def transformProblemForOptimizer(self, network):
        envMgr = EnvironmentVariableManager()
        cost = IsingPypsaInterface.buildCostFunction(
            network,
            float(envMgr["monetaryCostFactor"]),
            float(envMgr["kirchoffFactor"]),
            float(envMgr["minUpDownFactor"]),
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
        network = transformedProblem[0].addSQASolutionToNetwork(
            network, transformedProblem[0], solutionState
        )
        return network

    def optimize(self, transformedProblem):
        tic = time.perf_counter()
        result = self.solver.sample(transformedProblem[1])
        self.metaInfo["time"] = time.perf_counter() - tic
        self.metaInfo["energy"] = result.first.energy
        return result

    def getMetaInfo(self):
        return self.metaInfo


class DwaveTabuSampler(DwaveClassicalBackend):
    def __init__(self):
        self.solver = greedy.SteepestDescentSolver()
        self.metaInfo = {}


class DwaveCloudQuantumBackend(DwaveClassicalBackend):
    def __init__(self):
        envMgr = EnvironmentVariableManager()
        self.client = Client(token=envMgr["dwaveAPIToken"], solver="")
        self.solver = self.client.get_solver(envMgr["cloudSolverName"])
        self.metaInfo = {}
