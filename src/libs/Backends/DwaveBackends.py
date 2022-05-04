import time

from dwave.cloud import Client
import dimod
import tabu
import greedy

from .BackendBase import BackendBase
from .InputReader import InputReader
from .IsingPypsaInterface import IsingBackbone

from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite, EmbeddingComposite, DWaveCliqueSampler

from numpy import round as npround
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow import build_residual_network
from networkx.classes.function import set_edge_attributes

from os import path
from glob import glob
import mmap
import json

from pandas import value_counts


class DwaveTabuSampler(BackendBase):
    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        # TODO instert correct call for sampler
        self.sampler = None
        self.time = 0.0

    def processSolution(self, network, transformedProblem, solution):
        """
        gets and writes info about the solution and returns it as a dictionary.
        Does not improve the solution, instead override this method in child class
        for postprocessing.
        """
        bestSample = self.choose_sample(transformedProblem,
                                        solution,
                                        strategy=self.config["BackendConfig"]["strategy"]
                                        )

        resultInfo = transformedProblem.generateReport([
                id for id, value in bestSample.items() if value == -1
                ])
        self.output["results"] = {**self.output["results"], **resultInfo}
        return resultInfo

    def transformProblemForOptimizer(self, network):
        print("transforming Problem...")
        self.isingBackbone = IsingBackbone.buildIsingProblem(
                network,
                config=self.config["IsingInterface"]
                )
        return self.isingBackbone

    def getDimodModel(self, isingProblem: IsingBackbone) -> dimod.BinaryQuadraticModel:
        # store the directional qubits first, then the line's binary representations
        try:
            return self.dimodModel
        except AttributeError:
            linear = {
                spins[0]: strength
                for spins, strength in isingProblem.problem.items()
                if len(spins) == 1
            }
            # the convention is different to the sqa solver:
            # need to add a minus to the couplings
            quadratic = {
                spins: -strength
                for spins, strength in isingProblem.problem.items()
                if len(spins) == 2
            }
            self.dimodModel = dimod.BinaryQuadraticModel(
                            linear,
                            quadratic,
                            0,
                            dimod.Vartype.SPIN
                            )
            return self.dimodModel
    
    def getSampleDataframe(self):
        return self.sample_df

    def choose_sample(self,
                transformedProblem,
                solution,
                strategy="LowestEnergy"):
        df = self.getSampleDataframe()
        if strategy == 'LowestEnergy':
            return solution.first.sample

        # requires postprocessing because in order to match total power output
        # local knapsack problems are usually solved worsed compared to
        # the lowest energy sample
        if strategy == 'ClosestSample':
            totalLoad = 0.0
            for idx, _ in enumerate(self.network.snapshots):
                totalLoad += transformedProblem.getTotalLoad(idx)
            df['deviation_from_opt_load'] = df.apply(
                lambda row: abs(
                        totalLoad - \
                        transformedProblem.calcTotalPowerGenerated(
                        [id for id, value in row.items() if value == -1])
                        ),
                axis=1
            )
            min_deviation = df['deviation_from_opt_load'].min()
            ClosestSamples = df[df['deviation_from_opt_load'] == min_deviation]
            result_row = ClosestSamples.loc[ClosestSamples['energy'].idxmin()]
            return result_row[:-4].to_dict()

        # interpret strategy as index in sample df
        try:
            return df.iloc[strategy][:-4].to_dict()
        except TypeError:
            raise ValueError("The chosen strategy for picking a sample is not supported")

    def transformSolutionToNetwork(self, network, transformedProblem, solution):
        self.printReport()
        # network = transformedProblem.addSQASolutionToNetwork(
        #      network, solutionState
        # )
        return network

    def optimize(self, transformedProblem):
        print("starting optimization...")
        sampleset = self.getSampleSet(transformedProblem)
        self.saveSample(sampleset)
        print("done")
        return sampleset

    def getSampleSet(self, transformedProblem):
        return self.sampler.sample(self.getDimodModel(transformedProblem))

    def saveSample(self, sampleset):
        self.sample_df = sampleset.to_pandas_dataframe()
        self.output["results"]["sample_df"] = self.sample_df.to_dict('split')
        self.output["results"]["serial"] = sampleset.to_serializable()


class DwaveSteepestDescent(DwaveTabuSampler):
    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        self.solver = greedy.SteepestDescentSolver()


class DwaveCloud(DwaveTabuSampler):
    pass


class DwaveCloudHybrid(DwaveCloud):
    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        #self.token = self.envMgr["dwaveAPIToken"]
        self.token = self.config["APItoken"]["dWave_API_token"]
        self.solver = "hybrid_binary_quadratic_model_version2"
        self.sampler = LeapHybridSampler(solver=self.solver,
                                         token=self.token)
        self.output["results"]["solver_id"] = self.solver
        self.time = 0.0

    def getSampleSet(self, transformedProblem):
        sampleset = super().getSampleSet(transformedProblem)
        print("Waiting for server response...")
        # wait for response, ensure that loop is eventually broken
        watchdog = 0
        while True:
            if sampleset.done():
                break
            if watchdog > 42:
                raise ValueError
            time.sleep(2)
        return sampleset


class DwaveCloudDirectQPU(DwaveCloud):

    @staticmethod
    def get_filepaths(root_path: str, file_regex: str):
        return glob(path.join(root_path, file_regex))

    def validateInput(self, path, network):
        return

        # TODO implement file validation
        self.path = path
        self.networkName = network

        blacklists = self.get_filepaths(path, "*_qpu_blacklist")
        filteredByTimeout = []
        for blacklist in blacklists:
            blacklist_name = blacklist[len(path + "/"):]
            blacklisted_timeout = int(blacklist_name.split("_")[0])
            if blacklisted_timeout <= self.config["BackendConfig"]["timeout"]:
                filteredByTimeout.append(blacklist)

        for blacklist in filteredByTimeout:
            with open(blacklist) as f:
                s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                if s.find(bytes(network, 'utf-8')) != -1:
                    raise ValueError("network found in blacklist")

        embeddingPath = f'{path}/embedding_' \
                        f'rep_{self.config["IsingInterface"]["problemFormulation"]}_' \
                        f'{network}.json'
        if path.isfile(embeddingPath):
            print("found previous embedding")
            with open(embeddingPath) as embeddingFile:
                embedding = json.load(embeddingFile)

            self.embedding = {
                int(key): tuple(value)
                for key, value in embedding.items()
            }
        return

    def handleOptimizationStop(self, path, network):
        """
        If a network raises an error during optimization, add this network
        to the blacklisted networks for this optimizer and timeout value
        Blacklistfiles are of the form '{path}/{self.config["BackendConfig"]["timeout"]}_{Backend}_blacklist'
        """
        # on unix writing small buffers is atomic. no file locking necessary
        # append to existing file or create a new one
        with open(f'{path}/{self.config["BackendConfig"]["timeout"]}_qpu_blacklist', 'a+') as f:
            f.write(network + '\n')
        return

    def getSampler(self):
        DirectSampler = DWaveSampler(solver={
                                    'qpu': True,
                                    'topology__type': 'pegasus'
                                    },
                                    token=self.token)
        if hasattr(self, 'embedding'):
            self.sampler = FixedEmbeddingComposite(DirectSampler, self.embedding)
        else:
            self.sampler = EmbeddingComposite(DirectSampler)
        return self.sampler

    def getSampleSet(self, transformedProblem):
        sampleArguments = {
                    arg : val 
                    for arg, val in self.config["BackendConfig"].items() 
                    if arg in ["num_reads",
                            "annealing_time",
                            "chain_strength",
                            "programming_thermalization",
                            "readout_thermalization"]}
        if hasattr(self, 'embedding'):
            sampleArguments["embedding_parameters"] = dict(timeout=self.config["BackendConfig"]["timeout"])
            sampleArguments["return_embedding"] = True

        try:
            sampleset = self.sampler.sample(**sampleArguments)
        except ValueError:
            print("no embedding found in given time limit")
            raise ValueError("no embedding onto qpu was found")
        print("Waiting for server response...")
        while True:
            if sampleset.done():
                break
            time.sleep(1)
        return sampleset

    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        #self.token = self.envMgr["dwaveAPIToken"]
        self.token = self.config["APItoken"]["dWave_API_token"]
        # pegasus topology corresponds to Advantage 4.1
        self.getSampler()
        # additional info
        if self.config["BackendConfig"]["timeout"] < 0:
            self.config["BackendConfig"]["timeout"] = 3600

    def optimizeSampleFlow(self, sample, costKey):
        generatorState = [
            id for id, value in sample.items() if value == -1
        ]
        graph = self.buildFlowproblem(
            generatorState
        )
        return self.solveFlowproblem(
            graph,
            generatorState,
            costKey=costKey
        )

    def processSamples(self, transformedProblem, sampleset):
        processedSamples_df = sampleset.to_pandas_dataframe()
        processedSamples_df['quantumCost'] = processedSamples_df.apply(
            lambda row: transformedProblem.calcCost(
                    [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1
        )
        processedSamples_df['optimizedCost'] = processedSamples_df.apply(
            lambda row: self.optimizeSampleFlow(
                row[:-3],
                costKey=None,
            )[0],
            axis=1
        )
        processedSamples_df['marginalCost'] = processedSamples_df.apply(
            lambda row: transformedProblem.calcMarginalCost(
                    [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1
        )
        processedSamples_df['totalPower'] = processedSamples_df.apply(
            lambda row: transformedProblem.calcTotalPowerGenerated(
                    [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1
        )
        totalLoad = 0.0
        for idx, _ in enumerate(self.network.snapshots):
            totalLoad += transformedProblem.getTotalLoad(idx)
        processedSamples_df['deviation_from_opt_load'] = processedSamples_df.apply(
                lambda row: abs(
                                totalLoad - \
                                transformedProblem.calcTotalPowerGenerated(
                                        [id for id, value in row.items() if value == -1])
                                ),
                axis=1
        )
        return processedSamples_df[["energy",
                                    "quantumCost",
                                    "optimizedCost",
                                    "marginalCost",
                                    "totalPower",
                                    "deviation_from_opt_load"
                                    ]]


    def processSolution(self, network, transformedProblem, solution, sample=None):
        tic = time.perf_counter()
        resultDict = super().processSolution(
            network,
            transformedProblem,
            solution,
        )
        processedSamples_df = self.processSamples(transformedProblem, solution)
        lowestEnergyIndex = processedSamples_df["energy"].idxmin()
        self.output["results"]["LowestEnergy"] = processedSamples_df.iloc[lowestEnergyIndex]["quantumCost"]
        closestSamples = processedSamples_df[
                processedSamples_df['deviation_from_opt_load'] == \
                processedSamples_df['deviation_from_opt_load'].min()
                ]
        closestTotalPowerIndex = closestSamples['energy'].idxmin()
        self.output["samples_df"] = processedSamples_df.to_dict('index')
        self.output["results"]["lowestEnergy"] = processedSamples_df.iloc[lowestEnergyIndex]["quantumCost"]
        self.output["results"]["lowestEnergyProcessedFlow"] = processedSamples_df.iloc[lowestEnergyIndex]["optimizedCost"]
        self.output["results"]["closestPowerProcessedFlow"] = processedSamples_df.iloc[closestTotalPowerIndex]["optimizedCost"]
        self.output["results"]["bestProcessedFlow"] = processedSamples_df["optimizedCost"].min()
        self.output["results"]["postprocessingTime"] = time.perf_counter() - tic
        return resultDict

    def solveFlowproblem(self, graph, generatorState, costKey="totalCostFlow"):
        """
        solves the flow problem given in graph that corresponds to the
        generatorState in network. Calculates cost for a kirchhoffFactor of 1 
        and writes it to results["results"] under costKey. If costKey is None, it runs silently
        and only returns the computed cost value. 
        The generatorState is fixed, so if a costKey is given, the function only
        returns a dictionary of the values it computes for an optimal flow.
        The cost is written to the field in results["results"]. Flow solutions don't spread
        imbalances across all buses so they can still be improved. 
        """

        tic = time.perf_counter()
        FlowSolution = edmonds_karp(graph, "superSource", "superSink")
        toc = time.perf_counter()
        self.time += toc - tic

        # key errors occur iff there is no power generated or no load at a bus.
        # Power can still flow through the bus, but no cost is incurred
        totalCost = 0
        for bus in self.network.buses.index:
            try:
                totalCost += (FlowSolution['superSource'][bus]['capacity'] - \
                              FlowSolution['superSource'][bus]['flow']) ** 2
            except KeyError:
                pass
            try:
                totalCost += (FlowSolution[bus]['superSink']['capacity'] - \
                              FlowSolution[bus]['superSink']['flow']) ** 2
            except KeyError:
                pass

        if costKey is None:
            return totalCost, None

        print(f"total cost after flow opt {costKey}: {totalCost}")
        self.output["results"][costKey] = totalCost

        lineValues = {}
        # TODO split flow on two lines if we have more flow than capacity
        # we had lines a->b and b->a which were merged in previous step
        for line in self.network.lines.index:
            bus0 = self.network.lines.loc[line].bus0
            bus1 = self.network.lines.loc[line].bus1
            try:
                newValue = FlowSolution[bus0][bus1]['flow']
            except KeyError:
                try:
                    newValue = - FlowSolution[bus1][bus0]['flow']
                except KeyError:
                    newValue = 0
            lineValues[(line, 0)] = newValue
        return totalCost, lineValues

    # quantum computation struggles with finetuning powerflow to match
    # demand exactly. Using a classical approach to tune power flow can
    # archieved in polynomial time
    def buildFlowproblem(self, generatorState, lineValues=None, ):
        """
        build a self.networkx model to further optimise power flow. If using a warmstart,
        it uses the solution of the quantum computer encoded in generatorState to
        initialize a residual self.network. If the intial solution is good, a warmstart
        can speed up flow optimization by about 30%, but if it was bad, a warmstart
        makes it slower. warmstart is used if lineValues is not None
        """

        # turn pypsa self.network in nx.DiGraph. Power generation and consumption
        # is modeled by adjusting capacity of the edge to a super source/super sink 
        graph = nx.DiGraph()
        graph.add_nodes_from(self.network.buses.index)
        graph.add_nodes_from(["superSource", "superSink"])

        for line in self.network.lines.index:
            bus0 = self.network.lines.loc[line].bus0
            bus1 = self.network.lines.loc[line].bus1
            cap = self.network.lines.loc[line].s_nom
            # if self.network has multiple lines between buses, make sure not to 
            # erase the capacity of previous lines
            if graph.has_edge(bus0, bus1):
                graph[bus0][bus1]['capacity'] += cap
                graph[bus1][bus0]['capacity'] += cap
            else:
                graph.add_edges_from([(bus0, bus1, {'capacity': cap}),
                                      (bus1, bus0, {'capacity': cap})])

        for bus in self.network.buses.index:
            graph.add_edge(
                "superSource",
                bus,
                capacity=self.isingBackbone.calcTotalPowerGeneratedAtBus(bus, generatorState)
            )
        for load in self.network.loads.index:
            graph.add_edge(
                self.network.loads.loc[load].bus,
                "superSink",
                capacity=self.network.loads_t['p_set'].iloc[0][load],
            )
        # done building nx.DiGrpah

        if lineValues is not None:
            # generate flow for self.network lines
            for line in self.network.lines.index:
                bus0 = self.network.lines.loc[line].bus0
                bus1 = self.network.lines.loc[line].bus1
                if hasattr(graph[bus1][bus0], 'flow'):
                    graph[bus1][bus0]['flow'] -= lineValues[(line, 0)]
                else:
                    graph[bus0][bus1]['flow'] = lineValues[(line, 0)]

            # adjust source and sink flow to make it a valid flow. edges
            # to source/sink are not at full capacity iff there is a net
            # demand/power generated after subtraction power flow at that bus
            # might be wrong if kirchhoff constraint was violated in quantum
            # solution
            for bus in self.network.buses.index:
                generatedPower = self.isingBackbone.calcTotalPowerGeneratedAtBus(bus, generatorState)
                loadName = self.network.loads.index[self.network.loads.bus == bus][0]
                load = self.network.loads_t['p_set'].iloc[0][loadName]
                netFlowThroughBus = 0
                for line in self.network.lines.index[self.network.lines.bus0 == bus]:
                    netFlowThroughBus += lineValues[(line, 0)]
                for line in self.network.lines.index[self.network.lines.bus1 == bus]:
                    netFlowThroughBus -= lineValues[(line, 0)]
                netPower = generatedPower \
                           - load \
                           + netFlowThroughBus

                graph["superSource"][bus]['flow'] = min(generatedPower, generatedPower - netPower)
                graph[bus]["superSink"]['flow'] = min(load, load + netPower)

        return graph

    def optimize(self, transformedProblem):
        print("optimize")
        sampleset = self.getSampleSet(transformedProblem)
        self.sample_df = sampleset.to_pandas_dataframe()
        self.output["results"]["serial"] = sampleset.to_serializable()
        
        # TODO fix reusing embeddings
#        if not hasattr(self, 'embedding'):
#            embeddingPath = f'/energy/Problemset/embedding_' \
#                            f'{self.networkName}.json'
#
#            embeddingDict = self.output["results"]["serial"]["info"]["embedding_context"]["embedding"]
#            with open(embeddingPath, "w") as write_embedding:
#                json.dump(
#                    embeddingDict, write_embedding, indent=2
#                )
        self.output["results"]["optimizationTime"] = self.output["results"]["serial"]["info"]["timing"]["qpu_access_time"] / (10.0 ** 6)
        self.output["results"]["annealReadRatio"] = float(self.config["BackendConfig"]["annealing_time"]) / \
                                                    float(self.config["BackendConfig"]["num_reads"])
        self.output["results"]["totalAnnealTime"] = float(self.config["BackendConfig"]["annealing_time"]) * \
                                                    float(self.config["BackendConfig"]["num_reads"])
        # intentionally round totalAnnealTime so computations with similar anneal time
        # can ge grouped together
        self.output["results"]["mangledTotalAnnealTime"] = int(self.output["results"]["totalAnnealTime"] / 1000.0)
        return sampleset


class DwaveReadQPU(DwaveCloudDirectQPU):
    """
    This class behaves like it's parent except it doesn't
    use the Cloud. Instead it reads a serialized Sample and pretends
    that it got that from the cloud
    """

    def getSampler(self):
        self.inputFilePath =  "/energy/results_qpu/" +  self.config["BackendConfig"]["sampleOrigin"]

    def getSampleSet(self, transformedProblem):
        print(f"reading from {self.inputFilePath}")
        with open(self.inputFilePath) as inputFile:
            self.inputData = json.load(inputFile)
        if 'cutSamples' not in self.inputData:
            self.inputData['cutSamples'] = {}
        return dimod.SampleSet.from_serializable(self.inputData["serial"])

    def optimizeSampleFlow(self, sample, costKey):
        if costKey is None:
            try:
                previousResults = self.inputData["cutSamples"]
                thisResult = previousResults[sample.name]
                return thisResult, None
            except (KeyError, AttributeError):
                pass
        return super().optimizeSampleFlow(sample, costKey)
