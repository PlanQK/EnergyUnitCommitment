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
        self.time = 0.0
        #self.solver = tabu.Tabusampler()

    def validateInput(self, path, network):
        pass

    def handleOptimizationStop(self, path, network):
        pass

    def processSolution(self, network, transformedProblem, solution):
        """
        gets and writes info about the solution and returns it as a dictionary.
        Does not improve the solution, instead override this method in child class
        for postprocessing.
        """

        if hasattr(self, 'network'):
            bestSample = self.choose_sample(solution, self.network, strategy=self.config["DWaveBackend"]["strategy"])
        else:
            bestSample = self.choose_sample(solution, network)

        solutionState = [
            id for id, value in bestSample.items() if value == -1
        ]
        lineValues = transformedProblem[0].getLineValues(solutionState)
        indCostCon = transformedProblem[0].individualCostContribution(solutionState)
        totalCost = transformedProblem[0].calcCost(solutionState)
        self.output["results"]["marginalCost"] = transformedProblem[0].calcMarginalCost(solutionState)
        self.output["results"]["totalCost"] = totalCost
        self.output["results"]["kirchhoffCost"] = transformedProblem[0].calcKirchhoffCost(solutionState)
        print(f"Kirchhoff: {self.output['results']['kirchhoffCost']}")

        resultDict = {
            'solutionState': solutionState,
            'lineValues': lineValues,
            'individualCostContribution': indCostCon,
            'totalCost': totalCost,
        }
        return resultDict

    def transformProblemForOptimizer(self, network):
        print("transforming Problem...")

        cost = self.isingInterface

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
        """
        calculates total powerput of the network at a given snapshot.
        generatorState is given as list of index of active generators
        and not the qubit values of the solution.
        """
        result = 0
        num_generators = len(network.generators_t['p_max_pu'].loc[snapshot])
        for index in generatorState:
            if index >= num_generators:
                break
            result += network.generators_t['p_max_pu'].loc[snapshot].iloc[index]
        return result

    def choose_sample(self, solution, network, strategy="LowestEnergy", snapshot=None):

        if hasattr(self.output["results"], "sample_df"):
            df = self.output["results"]["sample_df"]
        else:
            df = solution.to_pandas_dataframe()
            self.output["results"]["sample_df"] = df.to_dict('split')

        if snapshot is None:
            snapshot = network.snapshots[0]

        if strategy == 'LowestEnergy':
            return solution.first.sample

        # basically always doesn't fulfill kirchhoff constraint and thus
        # yields bad results
        if strategy == 'MajorityVote':
            return df.mode().iloc[0]

        # instead of requireing a majority, a percentage voting for -1 is enough
        if strategy == 'PercentageVote':
            sample = df.apply(value_counts).fillna(0).apply(
                lambda col: float(col.loc[-1]) / float(len(df))
            )
            return sample.apply(
                lambda x: -1 if x >= self.config["DWaveBackend"]["threshold"] else 1
            )

        # requires postprocessing because in order to match total power output
        # local knapsack problems are usually solved worsed compared to
        # the lowest energy sample
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
            result_row = ClosestSamples.loc[ClosestSamples['energy'].idxmin()]
            return result_row[:-4].to_dict()

        # interpret strategy as index in sample df
        try:
            return df.iloc[strategy][:-4].to_dict()
        except TypeError:
            raise ValueError("The chosen strategy for picking a sample is not supported")

    def transformSolutionToNetwork(self, network, transformedProblem, solution):

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
        print(f"Marginal Cost: {transformedProblem[0].calcMarginalCost( solutionState)}")
        # network = transformedProblem.addSQASolutionToNetwork(
        #      network, solutionState
        # )
        return network

    def optimize(self, transformedProblem):
        print("starting optimization...")
        result = self.solver.sample(transformedProblem[1])
        self.output["results"]["serial"] = result.to_serializable()
        self.output["results"]["energy"] = result.first.energy
        print("done")
        return result


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

        # TODO write more info on solution to self.output["results"]

    def optimize(self, transformedProblem):
        print(self.sampler.properties)
        print("optimize")
        sampleset = self.sampler.sample(transformedProblem[1])
        print("Waiting for server response...")
        while True:
            if sampleset.done():
                break
            time.sleep(2)
        self.output["results"]["serial"] = sampleset.to_serializable()
        return sampleset


class DwaveCloudDirectQPU(DwaveCloud):

    @staticmethod
    def get_filepaths(root_path: str, file_regex: str):
        return glob(path.join(root_path, file_regex))

    def validateInput(self, networkPath, network):

        self.networkPath = networkPath
        self.networkName = network

        blacklists = self.get_filepaths(networkPath, "*_qpu_blacklist")
        filteredByTimeout = []
        for blacklist in blacklists:
            blacklist_name = blacklist[len(networkPath + "/"):]
            blacklisted_timeout = int(blacklist_name.split("_")[0])
            if blacklisted_timeout <= self.config["DWaveBackend"]["timeout"]:
                filteredByTimeout.append(blacklist)

        for blacklist in filteredByTimeout:
            with open(blacklist) as f:
                s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                if s.find(bytes(network, 'utf-8')) != -1:
                    raise ValueError("network found in blacklist")

        embeddingPath = f'{networkPath}/embedding_' \
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

    def handleOptimizationStop(self, networkPath, network):
        """
        If a network raises an error during optimization, add this network
        to the blacklisted networks for this optimizer and timeout value
        Blacklistfiles are of the form '{networkPath}/{self.config["DWaveBackend"]["timeout"]}_{Backend}_blacklist'
        """
        # on unix writing small buffers is atomic. no file locking necessary
        # append to existing file or create a new one
        with open(f'{networkPath}/{self.config["DWaveBackend"]["timeout"]}_qpu_blacklist', 'a+') as f:
            f.write(network + '\n')
        return

    def getSampler(self):
        self.sampler = EmbeddingComposite(
            DWaveSampler(
                solver={
                    'qpu': True,
                    'topology__type': 'pegasus'
                },
                token=self.token
            )
        )
        return

    def getSampleSet(self, transformedProblem):
        if hasattr(self, 'embedding'):
            print("Reusing embedding from previous run")
            sampler = DWaveSampler(solver={'qpu': True,
                                           'topology__type': 'pegasus'},
                                   token=self.token)
            sampler = FixedEmbeddingComposite(sampler, self.embedding)
            sampleset = sampler.sample(transformedProblem[1],
                                       num_reads=self.config["DWaveBackend"]["num_reads"],
                                       annealing_time=self.config["DWaveBackend"]["annealing_time"],
                                       chain_strength=self.config["DWaveBackend"]["chain_strength"],
                                       programming_thermalization=self.config["DWaveBackend"]["programming_thermalization"],
                                       readout_thermalization=self.config["DWaveBackend"]["readout_thermalization"],
                                       )
        else:
            try:
                sampleset = self.sampler.sample(transformedProblem[1],
                                                num_reads=self.config["DWaveBackend"]["num_reads"],
                                                annealing_time=self.config["DWaveBackend"]["annealing_time"],
                                                chain_strength=self.config["DWaveBackend"]["chain_strength"],
                                                programming_thermalization=self.config["DWaveBackend"]["programming_thermalization"],
                                                readout_thermalization=self.config["DWaveBackend"]["readout_thermalization"],
                                                embedding_parameters=dict(timeout=self.config["DWaveBackend"]["timeout"]),
                                                return_embedding=True,
                                                )
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
        if self.config["DWaveBackend"]["timeout"] < 0:
            self.config["DWaveBackend"]["timeout"] = 1000

        self.output["results"]["annealReadRatio"] = float(self.config["DWaveBackend"]["annealing_time"]) / \
                                                    float(self.config["DWaveBackend"]["num_reads"])
        self.output["results"]["totalAnnealTime"] = float(self.config["DWaveBackend"]["annealing_time"]) * \
                                                    float(self.config["DWaveBackend"]["num_reads"])
        # intentionally round totalAnnealTime so computations with similar anneal time
        # can ge grouped together
        self.output["results"]["mangledTotalAnnealTime"] = int(self.output["results"]["totalAnnealTime"] / 1000.0)

    def power_output(self, generatorState, snapshot):
        """
        calculate total power output of all all active generators at a
        snapshot. generatorState has to be the indices of generators.
        Indices that are out of range are ignored.
        """
        result = 0
        for index in generatorState:
            if index >= self.num_generators:
                break
            result += self.generators_t.loc[snapshot].iloc[index]
        return result

    def transformProblemForOptimizer(self, network):
        """
        stores some variables of the network that are necessary for later
        optimization in self. Also provides linesplitting of network to
        eliminate capacity constraint of lines. Then hands over actual
        ising formulation to parent class method
        """
        self.num_generators = len(network.generators_t['p_max_pu'].iloc[0])
        self.loads = {
            idx: network.loads_t['p_set'].loc[idx].sum()
            for idx in network.snapshots
        }
        self.generators_t = network.generators_t['p_max_pu']

        # copy that will be modified to better suit the problem. Only
        # self.network will be changed, the original network should 
        # never be changed
        self.network = network.copy()

        return super().transformProblemForOptimizer(self.network)

    def optimizeSampleFlow(self, sample, network, costKey):
        generatorState = [
            id for id, value in sample.items() if value == -1
        ]
        graph = self.buildFlowproblem(
            network,
            generatorState
        )
        return self.solveFlowproblem(
            graph,
            network,
            generatorState,
            costKey=costKey
        )

    def processSolution(self, network, transformedProblem, solution, sample=None):
        resultDict = super().processSolution(
            network,
            transformedProblem,
            solution,
        )

        lowestEnergySample = solution.first.sample
        lowestEnergyState = [
            id for id, value in lowestEnergySample.items() if value == -1
        ]
        self.output["results"]["LowestEnergy"] = transformedProblem[0].calcCost(lowestEnergyState)

        _, lineValuesLowestEnergyFlowSample = self.optimizeSampleFlow(
            lowestEnergySample,
            network,
            costKey="LowestFlow"
        )

        closestSample = self.choose_sample(solution, self.network, strategy="ClosestSample")
        _, lineValuesClosestSample = self.optimizeSampleFlow(
            closestSample,
            network,
            costKey="ClosestFlow"
        )


        # choose best self.config["DWaveBackend"]["sampleCutSize"] Samples and optimize Flow
        df = solution.to_pandas_dataframe()
        cutSamples = df.sort_values("energy", ascending=True).iloc[:self.config["DWaveBackend"]["sampleCutSize"]]
        cutSamples['quantumCost'] = cutSamples.apply(
            lambda row: transformedProblem[0].calcCost(
                    [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1
        )


        self.time = 0.0
        cutSamples['optimizedCost'] = cutSamples.apply(
            lambda row: self.optimizeSampleFlow(
                row[:-3],
                self.network,
                costKey=None,
            )[0],
            axis=1
        )
        self.output["results"]["postprocessingTime"] = self.time

        cutSamples['marginalCost'] = cutSamples.apply(
            lambda row: transformedProblem[0].calcMarginalCost(
                    [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1
        )




        self.output["results"]["cutSamples"] = cutSamples[[
                                                    "energy",
                                                    "quantumCost",
                                                    "optimizedCost",
                                                    'marginalCost'
                                                    ]].to_dict('index')
        self.output["results"]["cutSamplesCost"] = cutSamples['optimizedCost'].min()

        chosenSample = self.choose_sample(
                solution,
                self.network,
                strategy=self.config["DWaveBackend"]["strategy"]
                )
        print(chosenSample)
        self.output["results"]["marginalCost"] = transformedProblem[0].calcMarginalCost(
            [id for id, value in chosenSample.items() if value == -1]
        )
        self.time = 0.0
        self.optimizeSampleFlow(
            chosenSample,
            self.network,
            costKey="optimizedStrategySample"
        )

        print(f'cutSamplesCost with {self.config["DWaveBackend"]["sampleCutSize"]} samples: {self.output["results"]["cutSamplesCost"]}')

        if self.config["DWaveBackend"]["postprocess"] == "flow":
            if self.config["DWaveBackend"]["strategy"] == "LowestEnergy":
                resultDict['lineValues'] = lineValuesLowestEnergyFlowSample
            elif self.config["DWaveBackend"]["strategy"] == "ClosestSample":
                resultDict['lineValues'] = lineValuesClosestSample

        return resultDict

    def solveFlowproblem(self, graph, network, generatorState, costKey="totalCostFlow"):
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
        for bus in network.buses.index:
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

        print(f"TOTAL COST AFTER FLOW OPT {costKey}: {totalCost}")
        self.output["results"][costKey] = totalCost

        lineValues = {}
        # TODO split flow on two lines if we have more flow than capacity
        # we had lines a->b and b->a which were merged in previous step
        for line in network.lines.index:
            bus0 = network.lines.loc[line].bus0
            bus1 = network.lines.loc[line].bus1
            try:
                newValue = FlowSolution[bus0][bus1]['flow']
            except KeyError:
                try:
                    newValue = - FlowSolution[bus1][bus0]['flow']
                except KeyError:
                    newValue = 0
            lineValues[(line, 0)] = newValue
        return totalCost, lineValues

    def powerAtBus(self, network, generatorState, bus):
        """
        calculate generated power at a bus. generatorState has to be a list of indices
        of active generators. Indices that are out of range are ignored
        """
        activeGeneratorIndex = [idx for idx in generatorState if idx < self.num_generators]
        activeGenerators = network.generators_t['p_max_pu'].iloc[0].iloc[activeGeneratorIndex]
        return activeGenerators[network.generators.loc[activeGenerators.index].bus == bus].sum()

    # quantum computation struggles with finetuning powerflow to match
    # demand exactly. Using a classical approach to tune power flow can
    # archieved in polynomial time
    def buildFlowproblem(self, network, generatorState, lineValues=None, ):
        """
        build a networkx model to further optimise power flow. If using a warmstart,
        it uses the solution of the quantum computer encoded in generatorState to
        initialize a residual network. If the intial solution is good, a warmstart
        can speed up flow optimization by about 30%, but if it was bad, a warmstart
        makes it slower. warmstart is used if lineValues is not None
        """

        # turn pypsa network in nx.DiGraph. Power generation and consumption
        # is modeled by adjusting capacity of the edge to a super source/super sink 
        graph = nx.DiGraph()
        graph.add_nodes_from(network.buses.index)
        graph.add_nodes_from(["superSource", "superSink"])

        for line in network.lines.index:
            bus0 = network.lines.loc[line].bus0
            bus1 = network.lines.loc[line].bus1
            cap = network.lines.loc[line].s_nom
            # if network has multiple lines between buses, make sure not to 
            # erase the capacity of previous lines
            if graph.has_edge(bus0, bus1):
                graph[bus0][bus1]['capacity'] += cap
                graph[bus1][bus0]['capacity'] += cap
            else:
                graph.add_edges_from([(bus0, bus1, {'capacity': cap}),
                                      (bus1, bus0, {'capacity': cap})])

        for bus in network.buses.index:
            graph.add_edge(
                "superSource",
                bus,
                capacity=self.powerAtBus(network, generatorState, bus)
            )
        for load in network.loads.index:
            graph.add_edge(
                network.loads.loc[load].bus,
                "superSink",
                capacity=network.loads_t['p_set'].iloc[0][load],
            )
        # done building nx.DiGrpah

        if lineValues is not None:
            # generate flow for network lines
            for line in network.lines.index:
                bus0 = network.lines.loc[line].bus0
                bus1 = network.lines.loc[line].bus1
                if hasattr(graph[bus1][bus0], 'flow'):
                    graph[bus1][bus0]['flow'] -= lineValues[(line, 0)]
                else:
                    graph[bus0][bus1]['flow'] = lineValues[(line, 0)]

            # adjust source and sink flow to make it a valid flow. edges
            # to source/sink are not at full capacity iff there is a net
            # demand/power generated after subtraction power flow at that bus
            # might be wrong if kirchhoff constraint was violated in quantum
            # solution
            for bus in network.buses.index:
                generatedPower = self.powerAtBus(network, generatorState, bus)
                loadName = network.loads.index[network.loads.bus == bus][0]
                load = network.loads_t['p_set'].iloc[0][loadName]
                netFlowThroughBus = 0
                for line in network.lines.index[network.lines.bus0 == bus]:
                    netFlowThroughBus += lineValues[(line, 0)]
                for line in network.lines.index[network.lines.bus1 == bus]:
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

        self.output["results"]["serial"] = sampleset.to_serializable()

        if not hasattr(self, 'embedding'):
            embeddingPath = f'{self.networkPath}/embedding_' \
                            f'rep_{self.config["IsingInterface"]["problemFormulation"]}_' \
                            f'{self.networkName}.json'

            embeddingDict = self.output["results"]["serial"]["info"]["embedding_context"]["embedding"]
            with open(embeddingPath, "w") as write_embedding:
                json.dump(
                    embeddingDict, write_embedding, indent=2
                )
        self.output["results"]["optimizationTime"] = self.output["results"]["serial"]["info"]["timing"]["qpu_access_time"] / (10.0 ** 6)
        return sampleset


class DwaveReadQPU(DwaveCloudDirectQPU):
    """
    This class behaves like it's parent except it doesn't
    use the Cloud. Instead it reads a serialized Sample and pretends
    that it got that from the cloud
    """

    def getSampler(self):
        self.inputFilePath = self.config["DWaveBackend"]["sampleOrigin"]

    def getSampleSet(self, transformedProblem):

        with open(self.inputFilePath) as inputFile:
            self.inputData = json.load(inputFile)
        self.inputData = self.inputData["results"]
        if 'cutSamples' not in self.inputData:
            self.inputData['cutSamples'] = {}

        return dimod.SampleSet.from_serializable(self.inputData["serial"])

    def optimizeSampleFlow(self, sample, network, costKey):
        if costKey is None:
            try:
                previousResults = self.inputData["cutSamples"]
                thisResult = previousResults[sample.name]
                return thisResult, None
            except (KeyError, AttributeError):
                pass
        return super().optimizeSampleFlow(sample, network, costKey)

    def processSolution(self, network, transformedProblem, solution, sample=None):
        result = super().processSolution(network, transformedProblem, solution, sample)
        if len(self.inputData["cutSamples"]) < len(self.output["results"]["cutSamples"]):
            with open(self.inputFilePath, "w") as overwriteFile:
                json.dump(self.getResults(), overwriteFile, indent=2)
            print("adding more flow optimizations to qpu result file")
        return result
