import time

from dwave.cloud import Client
import dimod
import tabu
import greedy

from EnvironmentVariableManager import EnvironmentVariableManager
from .BackendBase import BackendBase
from .IsingPypsaInterface import IsingPypsaInterface

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


class DwaveTabuSampler(BackendBase):
    def __init__(self):
        self.solver = tabu.Tabusampler()
        self.metaInfo = {}

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

        if hasattr(self,'network'):
            bestSample = self.choose_sample(solution, self.network, strategy=self.strategy)
        else:
            bestSample = self.choose_sample(solution, network)

        solutionState = [
                id for id, value in bestSample.items() if value == -1
        ]
        lineValues = transformedProblem[0].getLineValues(solutionState)
        indCostCon = transformedProblem[0].individualCostContribution(solutionState)
        totalCost = transformedProblem[0].calcCost(solutionState)
        self.metaInfo["totalCost"] = totalCost

        resultDict = {
                'solutionState' : solutionState,
                'lineValues' : lineValues,
                'individualCostContribution' : indCostCon,
                'totalCost' : totalCost,
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
    
        if hasattr(self.metaInfo,"sample_df"):
            df = self.metaInfo["sample_df"]
        else:
            df = solution.to_pandas_dataframe()
            self.metaInfo["sample_df"] = df.to_dict('split')

        if snapshot is None:
            snapshot = network.snapshots[0]

        if strategy == 'LowestEnergy':
            return solution.first.sample

        # basically always doesn't fulfill kirchhoff constraint and thus
        # yields bad results
        if strategy == 'MajorityVote':
            return df.mode().iloc[0]

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
            return result_row[:-3]

        raise ValueError("The chosen strategy for picking a sample is not supported")


    @classmethod
    def transformSolutionToNetwork(cls, network, transformedProblem, solution):

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

        # TODO write more info on solution to metaInfo
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

    @staticmethod
    def get_filepaths(root_path: str, file_regex: str):
        return glob(path.join(root_path, file_regex))

    def validateInput(self, networkpath, network):
        
        self.networkPath = networkpath
        self.networkName = network

        blacklists = self.get_filepaths(networkpath, "*_qpu_blacklist")
        filteredByTimeout = []
        for blacklist in blacklists:
            blacklist_name = blacklist[len(networkpath + "/"):]
            blacklisted_timeout = int(blacklist_name.split("_")[0])
            if blacklisted_timeout <= self.timeout :
                filteredByTimeout.append(blacklist)

        for blacklist in filteredByTimeout:
            with open(blacklist) as f:
                s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                if s.find(bytes(network, 'utf-8')) != -1:
                    raise ValueError("network found in blacklist")

        embeddingPath=f"{networkpath}/embedding_gran_{self.granularity}_{network}.json"
        if path.isfile(embeddingPath):
            print("found previous embedding")
            with open(embeddingPath) as embeddingFile:
                embedding = json.load(embeddingFile)


            self.embedding = {
                    int(key):tuple(value) 
                    for key,value in embedding.items()
                    }
        return


    def handleOptimizationStop(self, networkpath, network):
        """
        If a network raises an error during optimization, add this network
        to the blacklisted networks for this optimizer and timeout value
        Blacklistfiles are of the form '{networkpath}/{self.timeout}_{Backend}_blacklist'
        """
        # on unix writing small buffers is atomic. no file locking necessary
        # append to existing file or create a new one
        with open(f"{networkpath}/{self.timeout}_qpu_blacklist" , 'a+') as f:
            f.write(network + '\n')
        return


    def __init__(self):
        envMgr = EnvironmentVariableManager()
        self.token = envMgr["dwaveAPIToken"]
        self.metaInfo = {}
        # pegasus topology corresponds to Advantage 4.1
        sampler = DWaveSampler(solver={'qpu' : True ,
                'topology__type': 'pegasus'},
                token=self.token)
        self.sampler = EmbeddingComposite(sampler)
             
        self.annealing_time = int(envMgr["annealing_time"])
        self.num_reads = int(envMgr["num_reads"])

        self.timeout = int(envMgr["timeout"])
        self.chain_strength = int(envMgr["chain_strength"])
        self.programming_thermalization = int(envMgr["programming_thermalization"])
        self.readout_thermalization = int(envMgr["readout_thermalization"])
        self.strategy = envMgr["strategy"]
        self.granularity = int(envMgr["granularity"])
        self.kirchhoffFactor = float(envMgr["kirchhoffFactor"])
        self.slackVarFactor = float(envMgr["slackVarFactor"])
        self.postprocess = envMgr["postprocess"]
        self.monetaryCostFactor = float(envMgr["monetaryCostFactor"])

        self.metaInfo["annealing_time"] = int(envMgr["annealing_time"])
        self.metaInfo["num_reads"] = int(envMgr["num_reads"])

        self.metaInfo["annealReadRatio"] = float(self.metaInfo["annealing_time"]) / \
                float(self.metaInfo["num_reads"])
        self.metaInfo["totalAnnealTime"] = float(self.metaInfo["annealing_time"]) * \
                float(self.metaInfo["num_reads"])
        # intentionally round totalAnnealTime imprecisely, so computations 
        # that have similar totatAnnealTime, but not exactly the same 
        # can be grouped together by plotting script
        self.metaInfo["mangledTotalAnnealTime"] = int(self.metaInfo["totalAnnealTime"] / 10.0)

        self.metaInfo["chain_strength"] = int(envMgr["chain_strength"])
        self.metaInfo["programming_thermalization"] = int(envMgr["programming_thermalization"])
        self.metaInfo["readout_thermalization"] = int(envMgr["readout_thermalization"])
        self.metaInfo["strategy"] = envMgr["strategy"]
        self.metaInfo["granularity"] = int(envMgr["granularity"])
        self.metaInfo["kirchhoffFactor"] = float(envMgr["kirchhoffFactor"])
        self.metaInfo["slackVarFactor"] = float(envMgr["slackVarFactor"])
        self.metaInfo["monetaryCostFactor"] = float(envMgr["monetaryCostFactor"])


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
                idx : network.loads_t['p_set'].loc[idx].sum() 
                for idx in network.snapshots
        } 
        self.generators_t = network.generators_t['p_max_pu']

        # copy that will be modified to better suit the problem. Only
        # self.network will be changed, the original network should 
        # never be changed
        self.network = network.copy()

        # line splitting. Stores data to retrieve original configuration in dict
        #keys: line name in original network
        #value: number of components of capacity 2^n-1 the decomposition is made
        #of. Line names in dummy network for an item (k,v) in the dictionary are
        #"{k}_split_{i}" for 0 <= i < v
        self.lineDictionary = {}
        self.lineNames = network.lines.index
        for line in self.lineNames:
            originalLine, numSplits = self.splitLine(line, self.network,self.granularity)
            self.lineDictionary[originalLine] = numSplits

        return super().transformProblemForOptimizer(self.network)


    def splitLine(self, line, network, granularity=0):
        """
        splits up a line into multiple lines such that each new line
        has capacity 2^n - 1 for some n. Modifies the network given
        as an argument and returns the original line name and how
        many components where used to split it up. Use it on the
        network stored in self because it modifies it.

        granularity is an upper limit for number of splits. if
        is is 0, no limit is set
        """
        remaining_s_nom = network.lines.loc[line].s_nom
        numComponents = 0
        while remaining_s_nom > 0 \
                and (granularity == 0 or granularity > numComponents):
            binLength = len("{0:b}".format(1+int(npround(remaining_s_nom))))-1
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
        """
        For a dictionary of lineValues of the network, whose
        lines were split up, uses the data in self to calculate
        the corresponding lineValues in the unmodified network
        """
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
    

    def processSolution(self, network, transformedProblem, solution, sample=None):
        resultDict = super().processSolution(
                network, 
                transformedProblem, 
                solution,
                )

        lineValues = self.mergeLines(resultDict['lineValues'], network.snapshots)
        resultDict['lineValues'] = lineValues

        # store lineValues of all other samples that were not chosen and save them
        # to metaInfo
        lineValuesDict = {
                idx : self.mergeLines(
                        transformedProblem[0].getLineValues(
                                self.metaInfo['sample_df']['data'][idx][:-3]
                        ),
                        network.snapshots,
                        )
                for idx in range(len(self.metaInfo['sample_df']['data']))
        }
        self.metaInfo['sample_df']['columns'].append('lineValues')
        for idx in range(len(self.metaInfo['sample_df']['data'])):
                self.metaInfo['sample_df']['data'][idx].append(
                        {
                                key[0]: value 
                                for key,value in lineValuesDict[idx].items()
                        }
            )


        lowestEnergySample = solution.first.sample
        lowestEnergyState = [
            id for id, value in lowestEnergySample.items() if value == -1
        ]
        self.metaInfo["LowestEnergy"] = transformedProblem[0].calcCost(lowestEnergyState)

        graphLowestEnergy = self.buildFlowproblem(network,
                lowestEnergyState,
                lineValues)
        lineValuesLowestEnergyFlowSample = self.solveFlowproblem(
                graphLowestEnergy,
                network,
                lowestEnergyState,
                costKey="LowestFlow"
        )

        closestSample = self.choose_sample(solution, self.network, strategy="ClosestSample")
        closestSampleState = [
            id for id, value in closestSample.items() if value == -1
        ]
        graphClosestSample = self.buildFlowproblem(network,
                closestSampleState,
                lineValues)
        lineValuesClosestSample = self.solveFlowproblem(
                graphClosestSample,
                network,
                closestSampleState,
                costKey="ClosestFlow"
        )


        if self.postprocess == "flow":
            if self.strategy == "LowestEnergy":
                resultDict['lineValues'] = lineValuesLowestEnergyFlowSample
            elif self.strategy == "ClosestSample":
                resultDict['lineValues'] = lineValuesClosestSample

        return resultDict


    def solveFlowproblem(self, graph, network, generatorState, costKey="totalCostFlow"):
        """
        solves the flow problem given in graph that corresponds to the
        generatorState in network. Calculates cost for a kirchhoffFactor of 1 
        and writes it to metaInfo. The generatorState is fixed, so it returns
        only a dictionary of the values it computes for an optimal flow. The flow
        solution doesn't spread imbalances across all buses so it can still be
        improved
        """
        FlowSolution = edmonds_karp(graph, "superSource", "superSink")

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

        print(f"TOTAL COST AFTER FLOW OPT {costKey}: {totalCost}")
        self.metaInfo[costKey] = totalCost

        lineValues = {} 
        # TODO split flow on two lines if we have more flow than capacity
        # we had lines a->b and b->a which were merged in previous step
        for line in network.lines.index:
            bus0=network.lines.loc[line].bus0
            bus1=network.lines.loc[line].bus1
            try:
                newValue = FlowSolution[bus0][bus1]['flow']
            except KeyError:
                try:
                    newValue = - FlowSolution[bus1][bus0]['flow']
                except KeyError:
                    newValue = 0
            lineValues[(line,0)] = newValue
        return lineValues


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
    def buildFlowproblem(self, network, generatorState, lineValues,warmstart=False):
        """
        build a networkx model to further optimise power flow. If using a warmstart,
        it uses the solution of the quantum computer encoded in generatorState to
        initialize a residual network. If the intial solution is good, a warmstart
        can speed up flow optimization by about 30%, but if it was bad, a warmstart
        makes it slower. 
        """

        # turn pypsa network in nx.DiGraph. Power generation and consumption
        # is modeled by adjusting capacity of the edge to a super source/super sink 
        graph = nx.DiGraph()
        graph.add_nodes_from(network.buses.index)
        graph.add_nodes_from(["superSource","superSink"])

        for line in network.lines.index:
            bus0 = network.lines.loc[line].bus0
            bus1 = network.lines.loc[line].bus1
            cap = network.lines.loc[line].s_nom
            # if network has multiple lines between buses, make sure not to 
            # erase the capacity of previous lines
            if graph.has_edge(bus0,bus1):
                graph[bus0][bus1]['capacity'] += cap
                graph[bus1][bus0]['capacity'] += cap
            else:
                graph.add_edges_from([(bus0,bus1,{'capacity': cap}),
                                (bus1,bus0,{'capacity': cap})])


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


        if warmstart:
            # generate flow for network lines
            for line in network.lines.index:
                bus0 = network.lines.loc[line].bus0
                bus1 = network.lines.loc[line].bus1
                if hasattr(graph[bus1][bus0],'flow'):
                    graph[bus1][bus0]['flow'] -= lineValues[(line,0)]
                else:
                    graph[bus0][bus1]['flow'] = lineValues[(line,0)]
            
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
                    netFlowThroughBus += lineValues[(line,0)]
                for line in network.lines.index[network.lines.bus1 == bus]:
                    netFlowThroughBus -= lineValues[(line,0)]
                netPower = generatedPower \
                        - load \
                        + netFlowThroughBus

                graph["superSource"][bus]['flow'] = min(generatedPower,generatedPower - netPower)
                graph[bus]["superSink"]['flow'] = min(load,load + netPower)

        return graph


    def optimize(self, transformedProblem):
        print("optimize")

        if hasattr(self,'embedding'):
            print("Reusing embedding from previous run")
            sampler = DWaveSampler(solver={'qpu' : True ,
                    'topology__type': 'pegasus'},
                    token=self.token)
            sampler = FixedEmbeddingComposite(sampler, self.embedding)
            sampleset = sampler.sample(transformedProblem[1],
                                num_reads=self.num_reads,
                                annealing_time=self.annealing_time,
                                chain_strength=self.chain_strength,
                                programming_thermalization=self.programming_thermalization,
                                readout_thermalization=self.readout_thermalization,
                                )
        else:
            try:
                sampleset = self.sampler.sample(transformedProblem[1],
                        num_reads=self.num_reads,
                        annealing_time=self.annealing_time,
                        chain_strength=self.chain_strength,
                        programming_thermalization=self.programming_thermalization,
                        readout_thermalization=self.readout_thermalization,
                        embedding_parameters=dict(timeout=self.timeout),
                        return_embedding = True,
                        )
            except ValueError:
                print("no embedding found in given time limit")
                raise ValueError("no embedding onto qpu was found")

        print("Waiting for server response...")
        while True:
            if sampleset.done():
                break
            time.sleep(1)

        self.metaInfo["serial"] = sampleset.to_serializable()

        if not hasattr(self,'embedding'):
            embeddingPath = f"{self.networkPath}/embedding_gran_{self.granularity}_{self.networkName}.json"
            embeddingDict = self.metaInfo["serial"]["info"]["embedding_context"]["embedding"]
            with open(embeddingPath, "w") as write_embedding:
                json.dump(
                    embeddingDict, write_embedding, indent=2
                )

        return sampleset
