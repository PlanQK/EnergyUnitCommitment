import time

from dwave.cloud import Client
import dimod
import tabu
import greedy

from EnvironmentVariableManager import EnvironmentVariableManager
from .BackendBase import BackendBase
from .IsingPypsaInterface import IsingPypsaInterface

from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, EmbeddingComposite


class DwaveTabuSampler(BackendBase):
    def __init__(self):
        self.solver = tabu.Tabusampler()
        self.metaInfo = {}

    def transformProblemForOptimizer(self, network):
        print("transforming Problem...")
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
        print("done")
        print(solutionState)
        print(transformedProblem[0].getLineValues(solutionState))
        print(transformedProblem[0].individualCostContribution(solutionState))
        print(
            f"Total cost (with constant terms): {transformedProblem[0].calcCost(solutionState)}"
        )

        network = transformedProblem[0].addSQASolutionToNetwork(
            network, transformedProblem[0], solutionState
        )
        return network

    def optimize(self, transformedProblem):
        print("starting optimization...")
        tic = time.perf_counter()
        result = self.solver.sample(transformedProblem[1])
        self.metaInfo["time"] = time.perf_counter() - tic
        self.metaInfo["energy"] = result.first.energy
        print("done")
        return result

    def getMetaInfo(self):
        return self.metaInfo


class DwaveSteepestDescent(DwaveTabuSampler):
    def __init__(self):
        self.solver = greedy.SteepestDescentSolver()
        self.metaInfo = {}


class DwaveCloud(DwaveTabuSampler):
    # set up all variables except the sampler
    def __init__(self):
        envMgr = EnvironmentVariableManager()
        self.token = envMgr["dwaveAPIToken"]
        self.metaInfo = {}


class DwaveCloudHybrid(DwaveCloud):
    def __init__(self):
        super().__init__()
        self.solver="hybrid_binary_quadratic_model_version2"
        self.sampler = LeapHybridSampler(solver=self.solver,
                token=self.token)

    def optimize(self, transformedProblem):
        print("optimize")

        sampleset = self.sampler.sample(transformedProblem[1])
        print("Waiting for server response...")
        while True:
            if sampleset.done():
                break
            time.sleep(3)

        #numpy's int32 are not serializable by json
        #conv_sample = {k:v.item() for (k,v) in sampleset.first.sample.items()}
        #self.metaInfo["status"] = conv_sample

        self.metaInfo["serial"] = sampleset.to_serializable()

        #sampler_info = ["qpu_access_time", "charge_time", "run_time"]
        #for info in sampler_info:
        #    if info in sampleset.info:
        #        self.metaInfo[info] = sampleset.info[info]
        return sampleset

class DwaveCloudDirectQPU(DwaveCloud):
    def __init__(self):
        super().__init__()
        sampler = DWaveSampler(solver={'qpu' : True},
                token=self.token)
        self.solver=sampler.solver.id
        self.metaInfo["solver_id"] = self.solver
        self.sampler = EmbeddingComposite(sampler)

    def optimize(self, transformedProblem):
        print("optimize")

        sampleset = self.sampler.sample(transformedProblem[1],
                num_reads = 4,
                annealing_time = 40
                #chain_strength=50
                )
        print("Waiting for server response...")
        while True:
            if sampleset.done():
                break
            time.sleep(3)

        #numpy's int32 are not serializable by json
        #conv_sample = {k:v.item() for (k,v) in sampleset.first.sample.items()}
        #self.metaInfo["status"] = conv_sample

        self.metaInfo["serial"] = sampleset.to_serializable()

        return sampleset
