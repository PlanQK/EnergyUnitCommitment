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

#import pandas
from numpy import round as npround 

class DwaveTabuSampler(BackendBase):
    def __init__(self):
        self.solver = tabu.Tabusampler()
        self.metaInfo = {}

    def processSolution(self, network, transformedProblem, solution):
        bestSample = self.choose_sample(solution, self.network, strategy=self.strategy)
        solutionState = [
            id for id, value in bestSample.items() if value == -1
        ]
        lineValues = transformedProblem[0].getLineValues(solutionState)
        indCostCon = transformedProblem[0].individualCostContribution(solutionState)
        totCost = transformedProblem[0].calcCost(solutionState)
        resultDict = {
                'solutionState' : solutionState,
                'lineValues' : lineValues,
                'individualCostContribution' : indCostCon,
                'totalCost' : totCost,
        }
        return resultDict

    def transformProblemForOptimizer(self, network):
        print("transforming Problem...")

        cost = IsingPypsaInterface.buildCostFunction(
            network,
        )

        
        # store the directional qubits first, then the line's binary representations
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
    def power_output(network, generatorState, snapshot):
        result = 0
        num_generators = len(network.generators_t['p_max_pu'].loc[snapshot])
        for index in generatorState:
            if index >= num_generators:
                break
            result += network.generators_t['p_max_pu'].loc[snapshot].iloc[index]
        return result


    @staticmethod
    def choose_sample(solution, network, strategy="LowestEnergy", snapshot=None):
    
        if strategy == 'LowestEnergy':
            return solution.first.sample

        if snapshot is None:
            snapshot = network.snapshots[0]

        df = solution.to_pandas_dataframe()
        print(f"INDEX LOWEST E: {df['energy'].idxmin()}\n")

        if strategy == 'MajorityVote':
            return df.mode().iloc[0]

        if strategy == 'ClosestSample':
            total_load = network.loads_t['p_set'].loc[snapshot].sum()
            df['deviation_from_opt_load'] = df.apply(
                    lambda row: abs(total_load -
                                    DwaveTabuSampler.power_output(
                                        network,
                                        [id for id, value in row.items() if value == -1],
                                        snapshot
                                        )
                                    ),
                    axis=1
            )
            min_deviation = df['deviation_from_opt_load'].min()
            ClosestSamples = df[df['deviation_from_opt_load'] == min_deviation]
            print(f"INDEX OF CHOSEN SAMPLE: {ClosestSamples['energy'].idxmin()}")
            result_row = ClosestSamples.loc[ClosestSamples['energy'].idxmin()]
            return result_row[:-3]

        raise ValueError("The chosen strategy for picking a sample is not supported")

#    @staticmethod
#    def choose_sample(solution, network):
#        return solution.first.sample

    @classmethod
    def transformSolutionToNetwork(cls, network, transformedProblem, solution):
#        bestSample = cls.choose_sample(solution, network, strategy='LowestEnergy')
#        solutionState = [
#            id for id, value in bestSample.items() if value == -1
#        ]

        solutionState = solution['solutionState']
        
        print("done")
        print(solutionState)
        print(solution['lineValues'])
        print(solution['individualCostContribution'])
        print(
            f"Total cost (with constant terms): {solution['totalCost']}"
        )
        for snapshot in network.snapshots:
            power = DwaveTabuSampler.power_output(network,
                        solutionState,
                        snapshot)
            load = network.loads_t['p_set'].loc[snapshot].sum()
            print(f"Total output at {snapshot}: {power}")
            print(f"Total load at {snapshot}: {load}")

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
    pass


class DwaveCloudHybrid(DwaveCloud):
    def __init__(self):
        envMgr = EnvironmentVariableManager()
        self.token = envMgr["dwaveAPIToken"]
        self.metaInfo = {}
        self.solver="hybrid_binary_quadratic_model_version2"
        self.sampler = LeapHybridSampler(solver=self.solver,
                token=self.token)
        self.metaInfo["solver_id"] = self.solver

    def optimize(self, transformedProblem):
        print("optimize")
        sampleset = self.sampler.sample(transformedProblem[1])
        print("Waiting for server response...")
        while True:
            if sampleset.done():
                break
            time.sleep(2)
        self.metaInfo["serial"] = sampleset.to_serializable()
        return sampleset


class DwaveCloudDirectQPU(DwaveCloud):
    def __init__(self):
        envMgr = EnvironmentVariableManager()
        self.token = envMgr["dwaveAPIToken"]
        self.metaInfo = {}
        # pegasus topology corresponds to Advantage 4.1
        sampler = DWaveSampler(solver={'qpu' : True ,
                'topology__type': 'pegasus'},
                token=self.token)
        self.solver=sampler.solver.id
        self.metaInfo["solver_id"] = self.solver
        self.sampler = EmbeddingComposite(sampler)
        self.annealing_time = int(envMgr["annealing_time"])
        self.num_reads = int(envMgr["num_reads"])
        self.chain_strength = int(envMgr["chain_strength"])
        self.programming_thermalization = int(envMgr["programming_thermalization"])
        self.readout_thermalization = int(envMgr["readout_thermalization"])
        self.strategy = envMgr["strategy"]

    def power_output(self, generatorState, snapshot):
        result = 0
        for index in generatorState:
            if index >= self.num_generators:
                break
            result += self.generators_t.loc[snapshot].iloc[index]
        return result

    def transformProblemForOptimizer(self, network):
        self.num_generators = len(network.generators_t['p_max_pu'].iloc[0])
        self.loads = {
                idx : network.loads_t['p_set'].loc[idx].sum() 
                for idx in network.snapshots
        } 
        self.generators_t = network.generators_t['p_max_pu']
        self.network = network.copy()

        self.lineDictionary = {}
        self.lineNames = network.lines.index

        for line in self.lineNames:
            k, v = self.splitLine(line, self.network)
            self.lineDictionary[k]=v

        return super().transformProblemForOptimizer(self.network)

    def splitLine(self, line, network):
        """
        splits up a line into multiple lines such that each new line
        has capacity 2^n - 1 for some n. Modifies the network given
        as an argument and returns the original line name and how
        many components where used to split it up
        """
        remaining_s_nom = network.lines.loc[line].s_nom
        componentNames = []
        numComponents = 0
        while remaining_s_nom > 0:
            binLength = len("{0:b}".format(int(npround(remaining_s_nom))))-1
            if binLength == 0:
                binLength += 1
            magnitude = 2 ** binLength - 1
            newName = f"{line}_split_{numComponents}"
            remaining_s_nom -= magnitude
            componentNames.append(newName)
            network.add(
                "Line",
                newName,
                bus0=network.lines.loc[line].bus0,
                bus1=network.lines.loc[line].bus1,
                s_nom=magnitude
            )
            numComponents += 1
        network.remove("Line",line)
        return (line, numComponents)

    def mergeLines(self, lineValues, snapshots):
        result = {}
        for line, numSplits in self.lineDictionary.items():
            for snapshot in range(len(snapshots)):
                newKey = (line, snapshot)
                value = 0
                for i in range(numSplits):
                    splitLineKey = line + "_split_" + str(i)
                    value += lineValues[(splitLineKey, snapshot)]
                result[newKey] = value
        return result
    
#        lineValues = transformedProblem[0].getLineValues(solutionState)
#        indCostCon = transformedProblem[0].individualCostContribution(solutionState)
#        totCost = transformedProblem[0].calcCost(solutionState)
#        resultDict = {
#                'solutionState' : solutionState,
#                'lineValues' : lineValues,
#                'individualCostContribution' : indCostCon,
#                'totalCost' : totCost,

    def processSolution(self, network, transformedProblem, solution, sample=None):
        bestSample = self.choose_sample(solution, self.network)
        resultDict = super().processSolution(
                network, 
                transformedProblem, 
                solution,
                )

        lineValues = self.mergeLines(resultDict['lineValues'], network.snapshots)
        resultDict['lineValues'] = lineValues

        # optimize flow using classical greedy algorithms



        return resultDict

    def optimize(self, transformedProblem):
        print("optimize")
        sampleset = self.sampler.sample(transformedProblem[1],
                num_reads=self.num_reads,
                annealing_time=self.annealing_time,
                chain_strength=self.chain_strength,
                programming_thermalization=self.programming_thermalization,
                readout_thermalization=self.readout_thermalization,
                )
        print("Waiting for server response...")
        while True:
            if sampleset.done():
                break
            time.sleep(2)

        self.metaInfo["serial"] = sampleset.to_serializable()

        return sampleset
