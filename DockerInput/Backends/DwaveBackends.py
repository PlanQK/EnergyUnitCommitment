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

from numpy import round as npround 
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow import build_residual_network
from networkx.classes.function import set_edge_attributes


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
        print(df.head())
        print("...")
        print(df.iloc[df['energy'].idxmin()])

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

        #keys: line name in original network
        #value: number components of capacity 2^n-1 the decomposition is made of
        #line names in dummy network for an item (k,v) in the dictionary are
        #"{k}_split_{i}" for 0 <= i < v
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
        many components where used to split it up. Use it on the
        network stored in self.
        """
        remaining_s_nom = network.lines.loc[line].s_nom
        numComponents = 0
        while remaining_s_nom > 0:
            binLength = len("{0:b}".format(int(npround(remaining_s_nom))))-1
            #case remaining_s_nom <= 1
            if binLength == 0: 
                binLength += 1
            magnitude = 2 ** binLength - 1
            remaining_s_nom -= magnitude
            network.add(
                "Line",
                f"{line}_split_{numComponents}",
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
        resultDict = super().processSolution(
                network, 
                transformedProblem, 
                solution,
                )

        lineValues = self.mergeLines(resultDict['lineValues'], network.snapshots)
        resultDict['lineValues'] = lineValues

        # optimize flow using classical algorithms

        G,R = self.buildFlowproblem(network,
                resultDict['solutionState'],
                lineValues)

        tic = time.perf_counter()
        Other = edmonds_karp(G, "superSource", "superSink")
        self.metaInfo["max_flow_time_without_quantum_start"] = time.perf_counter()-tic
        print(Other.edges(data=True))

#        Opt = edmonds_karp(G, "superSource", "superSink", residual = R)
#        print("Opt")
#        print(Opt.edges(data=True))

        return resultDict

    def powerAtBus(self, network, generatorState, bus):
        activeGeneratorIndex = [idx for idx in generatorState if idx < self.num_generators]
        activeGenerators = network.generators_t['p_max_pu'].iloc[0].iloc[activeGeneratorIndex]
        return activeGenerators[network.generators.loc[activeGenerators.index].bus == bus].sum()


    # the quantum computation struggles with finetuning powerflow to match
    # demand exactly due to flipping single bits corresponding to a large 
    # change in power. we hope that using a classical approach to tune is
    # practically free due to our solution already being very close to the
    # global minimum
    def buildFlowproblem(self, network, generatorState, lineValues):
        """
        build a networkx model to further optimise power flow with a residual
        network corresponding to the quantum solution.
        """

        # turn pypsa network in nx.DiGraph. Power generation and consumption
        # is modeled by adjusting capacity of the edge to a super source/super sink 
        G = nx.DiGraph()
        G.add_nodes_from(network.buses.index)
        G.add_nodes_from(["superSource","superSink"])

        for line in network.lines.index:
            bus0 = network.lines.loc[line].bus0
            bus1 = network.lines.loc[line].bus1
            cap = network.lines.loc[line].s_nom
            # network has two lines between buses, make sure not to 
            # erase the capacity of previous line
            if G.has_edge(bus0,bus1):
                G[bus0][bus1]['capacity'] += cap
                G[bus1][bus0]['capacity'] += cap
            else:
                G.add_edges_from([(bus0,bus1,{'capacity': cap}),
                                (bus1,bus0,{'capacity': cap})])

        for bus in network.buses.index:
            G.add_edge(
                "superSource",
                bus,
                capacity=self.powerAtBus(network, generatorState, bus)
            )

        for load in network.loads.index:
            G.add_edge(
                network.loads.loc[load].bus,
                "superSink",
                capacity=network.loads_t['p_set'].iloc[0][load],
            )
        # done building nx.DiGrpah

        R = build_residual_network(G,capacity=511)
        set_edge_attributes(R,0,'flow')

        # generate flow for network lines
        for line in network.lines.index:
            bus0 = network.lines.loc[line].bus0
            bus1 = network.lines.loc[line].bus1
            R[bus0][bus1]['flow'] += lineValues[(line,0)]
            R[bus1][bus0]['flow'] -= lineValues[(line,0)]

        # adjust source and sink flow to make it a valid residual network
        # edges to source/sink are not at full capacity iff there is a net
        # demand/power generated after subtraction power flow at that bus
        for bus in network.buses.index:
            generatedPower = self.powerAtBus(network, generatorState, bus)
            loadName = network.loads.index[network.loads.bus == bus][0]
            load = network.loads_t['p_set'].iloc[0][loadName]
            netFlowThroughBus = 0
            for line in network.lines.index[network.lines.bus0 == bus]:
                netFlowThroughBus += lineValues[(line,0)]
            for line in network.lines.index[network.lines.bus1 == bus]:
                netFlowThroughBus -= lineValues[(line,0)]
            netPower = generatedPower \
                    - load \
                    + netFlowThroughBus

            R["superSource"][bus]['flow'] = min(generatedPower,generatedPower-netPower)
            R[bus]["superSource"]['flow'] = -R["superSource"][bus]['flow']
            R[bus]["superSink"]['flow'] = min(load,load + netPower)
            R["superSink"][bus]['flow'] = -R[bus]["superSink"]['flow']

        return G,R


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
