import time

# importing dwave packages
import pandas
from dwave.cloud import Client
import dimod
import tabu
import greedy

from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite, \
    EmbeddingComposite, DWaveCliqueSampler
from tabu import TabuSampler

# importing local QUBO modelling packages
from .BackendBase import BackendBase
from .InputReader import InputReader
from .IsingPypsaInterface import IsingBackbone

# import packages for flow optimization
from numpy import round as npround
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow import build_residual_network
from networkx.classes.function import set_edge_attributes

# import packages for reading files
from os import path
from glob import glob
import mmap
import json

from pandas import value_counts


class DwaveTabuSampler(BackendBase):
    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        self.getSampler()

    def getSampler(self):
        """
        Returns the D-Wave sampler and stores it as the attribute
        sampler.
        This method will be overridden in subclasses by chooosing
        different samplers.

        Returns:
            (Sampler)
                The optimizer that generates samples of solutions.
        """
        self.sampler = TabuSampler()
        return self.sampler

    def processSamples(self, sampleset) -> pandas.Dataframe:
        """
        processes a returned sample set and constructs a pandas
        dataframe with all available information of that set.
        If necessary, this method will also add additonal columns
        containing derived values.

        Args:
            sampleset:
                The sampleset that is returned by the D-Wave solver.
        Returns:
            (pandas.DataFrame)
                A DataFrame containing all relevant data of the samples.
        """
        processedSamples_df = sampleset.to_pandas_dataframe()
        processedSamples_df['isingCost'] = processedSamples_df.apply(
            lambda row: self.transformedProblem.calcCost(
                [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1
        )
        processedSamples_df['marginalCost'] = processedSamples_df.apply(
            lambda row: self.transformedProblem.calcMarginalCost(
                [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1
        )
        processedSamples_df['totalPower'] = processedSamples_df.apply(
            lambda row: self.transformedProblem.calcTotalPowerGenerated(
                [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1
        )
        return processedSamples_df

    def processSolution(self):
        """
        Gets and writes info about the sample_df and returns it as a
        dictionary.
        """
        bestSample = self.choose_sample()
        resultInfo = self.transformedProblem.generateReport([
            id for id, value in bestSample.items() if value == -1
        ])
        self.output["results"] = {**self.output["results"], **resultInfo}

    def transformProblemForOptimizer(self):
        """
        Initializes an IsingInterface-instance, which encodes the Ising
        Spin Glass Problem, using the network to be optimized.

        Returns:
            (IsingBackbone)
                The IsingInterface-instance, which encodes the Ising
                Spin Glass Problem.
        """
        print("transforming Problem...")
        self.isingBackbone = IsingBackbone.buildIsingProblem(
            network=self.network,
            config=self.config["IsingInterface"]
        )
        return self.isingBackbone

    def getDimodModel(self, isingProblem: IsingBackbone) \
            -> dimod.BinaryQuadraticModel:
        """
        For a ising spin glass model given through an IsingBackbone
        object, returns the corresponding D-Wave
        dimod.BinaryQuadraticModel.
    
        Args:
            isingProblem: (IsingBackbone)
                The IsingBackbone that contains the problem that is to
                be transformed into a D-Wave BinaryQuadraticModel.
        Returns:
            (dimod.BinaryQuadraticModel)
                The equivalent D-Wave model of the IsingBackbone.

        """
        # store the directional qubits first, then the line's binary
        # representations
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
        """
        Returns the data frame containing the data of the samples and
        their derived data.
    
        Returns:
            (pandas.DataFrame)
                The data frame containing a sample in each row.
        """
        return self.sample_df

    def choose_sample(self):
        """
        After sampling a QUBO this function chooses one sample to be
        returned as the solution.
        Since the sampler of this class only returns one sample,
        choosing a solution is trivial. This function is overwritten in
        subclasses that return more than one sample.

        Returns:
            (pandas.Series)
                The chosen row to be used as the solution

        """
        return self.getSampleDataframe().iloc[0]

    #
    def transformSolutionToNetwork(self) -> None:
        """
        Encodes the optimal solution found during optimization in a
        pypsa.Network and stores it in self.output. It reads the
        solution stored in the optimizer instance, prints some
        information about it and then writes it to the network.

        Returns:
            (None)
                Stores the outputNetwork as dicitonary in self.output.
        """
        self.printReport()

        # TODO: check choose_sample function
        bestSample = self.choose_sample()

        outputNetwork = self.transformedProblem.setOutputNetwork(solution=[
            id for id, value in bestSample.items() if value == -1])
        outputDataset = outputNetwork.export_to_netcdf()
        self.output["network"] = outputDataset.to_dict()

    def optimize(self) -> None:
        """
        Optimizes the problem encoded in the IsingBackbone-Instance
        using Tabu search.

        Returns:
            (None)
                The optimized solution is stored in self.output.
        """
        print("starting optimization...")
        sampleset = self.getSampleSet()
        self.sample_df = self.processSamples(sampleset)
        self.saveSample(sampleset)
        print("done")

    def getSampleSet(self):
        """
        Queries the sampler to sample the problem and return all
        solutions.
        Overwriting this method allows alternative ways to get samples
        that contain solutions. This is especially useful for reading
        serialized samples from disk.

        Returns:
            (SampleSet)
                A record of the samples and any data returned by the
                sampler.

        """
        return self.sampler.sample(self.getDimodModel(self.transformedProblem))

    def saveSample(self, sampleset):
        """
        Saves the sampleset as a data frame to be used by other methods
        and in the output dictionary since it contains all solutions
        found by the solve.
    
        Args:
            sampleset: (SampleSet)
                Record of all samples and additional data on them.
        Returns:
            (None)
                Modifies `self.sample_df` and `self.output`.
        """
        if not hasattr(self, "sample_df"):
            self.sample_df = sampleset.to_pandas_dataframe()
        self.output["results"]["sample_df"] = self.sample_df.to_dict('split')
        self.output["results"]["serial"] = sampleset.to_serializable()


class DwaveSteepestDescent(DwaveTabuSampler):
    # TODO test if this actually runs
    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        self.solver = greedy.SteepestDescentSolver()


class DwaveCloud(DwaveTabuSampler):
    """
    Class for structuring the class hierachy. Any class that use results
    obtained by querying D-Wave servers should inherit from this class.
    """


class DwaveCloudHybrid(DwaveCloud):
    def getSampler(self):
        """
        Returns a D-Wave sampler that will query the hybrid solver for
        solving the Ising problem.
        In order to appropriately configure the solver, this method will
        also read the config attribute and save some settings as
        attributes.

        Returns:
            (Sampler)
                The optimizer that generates samples of solutions.
        """
        self.token = self.config["APItoken"]["dWave_API_token"]
        self.solver = "hybrid_binary_quadratic_model_version2"
        self.sampler = LeapHybridSampler(solver=self.solver,
                                         token=self.token)
        self.output["results"]["solver_id"] = self.solver

    def getSampleSet(self):
        """
        A method for obtaining a solution from d-wave using their hybrid
        solver.
        Since this queries the d-wave servers, it will have to wait for
        an answer which might take a while. Sampling will not abort if
        the response takes too long.

        Returns:
            (SampleSet)
                The sampleset containing a solution.
        """
        sampleset = super().getSampleSet()
        print("Waiting for server response...")
        # wait for response, no safeguard for endless looping
        watchdog = 0
        while True:
            if sampleset.done():
                break
            time.sleep(2)
        return sampleset


class DwaveCloudDirectQPU(DwaveCloud):

    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        if self.config["BackendConfig"]["timeout"] < 0:
            self.config["BackendConfig"]["timeout"] = 3600

    @staticmethod
    def get_filepaths(root_path: str, file_regex: str):
        return glob(path.join(root_path, file_regex))

    def validateInput(self, path):
        return

        self.path = path
        self.networkName = ""

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
                if s.find(bytes(networkName, 'utf-8')) != -1:
                    raise ValueError("network found in blacklist")

        embeddingPath = (
            f'{path}/embedding_rep_'
            f'{self.config["IsingInterface"]["problemFormulation"]}_'
            f'{networkName}.json')
        if path.isfile(embeddingPath):
            print("found previous embedding")
            with open(embeddingPath) as embeddingFile:
                embedding = json.load(embeddingFile)

            self.embedding = {
                int(key): tuple(value)
                for key, value in embedding.items()
            }
        return

    def handleOptimizationStop(self, path):
        """
        If a network raises an error during optimization, add this
        network to the blacklisted networks for this optimizer and
        timeout value Blacklistfiles are of the form
        '{path}/{self.config["BackendConfig"]["timeout"]}'
        '_{Backend}_blacklist'
        """
        # on unix writing small buffers is atomic. no file locking necessary
        # append to existing file or create a new one
        # with open(f'{path}/{self.config["BackendConfig"]["timeout"]}_'
        #          f'qpu_blacklist', 'a+') as f:
        #    f.write(network + '\n')
        return

    def getSampler(self):
        """
        Returns a D-Wave sampler that will query the quantum annealer
        (pegasus topology) for solving the ising problem.
        In order to appropriately configure the solver, this method will
        also read the config attribute and save some settings as
        attributes. If a fitting embedding is found, it will be reused
        to embed the problem onto the hardware.

        Returns:
            (Sampler)
                The optimizer that generates samples of quantum
                annealing runs.
        """
        self.token = self.config["APItoken"]["dWave_API_token"]
        # pegasus topology corresponds to Advantage 4.1
        DirectSampler = DWaveSampler(solver={
            'qpu': True,
            'topology__type': 'pegasus'
        },
            token=self.token)
        if hasattr(self, 'embedding'):
            self.sampler = FixedEmbeddingComposite(DirectSampler,
                                                   self.embedding)
        else:
            self.sampler = EmbeddingComposite(DirectSampler)
        return self.sampler

    def getSampleSet(self):
        """
        Queries the quantum annealer to sample the problem and return
        all solutions.
        Parameters for configuring the quantum annealer are read from
        the `config` attribute.

        Returns:
            (SampleSet)
                A record of the quantum annealing samples returned by
                the sampler.
        """
        # construct annealer arguments. necessary to differentiate if you are
        # using a preexisting embedding or have to find one for this run
        sampleArguments = {
            arg: val
            for arg, val in self.config["BackendConfig"].items()
            if arg in ["num_reads",
                       "annealing_time",
                       "chain_strength",
                       "programming_thermalization",
                       "readout_thermalization"]}
        if hasattr(self, 'embedding'):
            sampleArguments["embedding_parameters"] = dict(
                timeout=self.config["BackendConfig"]["timeout"])
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

    def optimizeSampleFlow(self, sample):
        """
        A method for postprocessing a quantum annealing sample.
        Since current quantum annealers are very noisy and chains make
        finding solutions harder, most solutions will have errors in
        their qubit states leading to bad solutions. In order to improve
        the solutions without losing any potential quantum advantage,
        we optimize the power flow after using quantum annealing since
        this problem can be solved in polynomial time.

        Args:
            sample: (pandas.Series)
                The row of the sample data frame.
        Returns:
            (int)
                The kirchhoffCost of an optimized power flow using the
                generators committed according to the given sample.
        """
        generatorState = [
            id for id, value in sample.items() if value == -1
        ]
        graph = self.buildFlowproblem(
            generatorState
        )
        return self.solveFlowproblem(
            graph,
            generatorState,
        )

    def processSamples(self, sampleset):
        """
        A method for turning a sampleset into a data frame and add
        additional data.
        The method transforms the sampleset into the corresponding data
        frame. Then this class and subclasses overriding this method use
        pandas to add new columns containing additional information
        about each sample.

        Args:
            sampleset: (SampleSet)
                A d-wave sampleset containing the annealing data.
        Returns:
            (pandas.DataFrame)
                The data frame of the sample set with extra columns.
        """
        processedSamples_df = super().processSamples(sampleset)
        processedSamples_df['optimizedCost'] = processedSamples_df.apply(
            lambda row: self.optimizeSampleFlow(
                row[:-3],
            ),
            axis=1
        )
        totalLoad = 0.0
        for idx, _ in enumerate(self.network.snapshots):
            totalLoad += self.transformedProblem.getTotalLoad(idx)
        processedSamples_df['deviation_from_opt_load'] = \
            processedSamples_df.apply(
                lambda row: abs(
                    totalLoad
                    - self.transformedProblem.calcTotalPowerGenerated(
                        [id for id, value in row.items() if value == -1]
                    )
                ), axis=1
            )
        return processedSamples_df

    def processSolution(self, network, sample_df):
        tic = time.perf_counter()
        super().processSolution(
            network,
            sample_df,
        )
        lowestEnergyIndex = sample_df["energy"].idxmin()
        self.output["results"]["LowestEnergy"] = sample_df.iloc[
            lowestEnergyIndex]["isingCost"]
        closestSamples = sample_df[
            sample_df['deviation_from_opt_load'] == \
            sample_df['deviation_from_opt_load'].min()
            ]
        closestTotalPowerIndex = closestSamples['energy'].idxmin()
        self.output["samples_df"] = sample_df.to_dict('index')
        self.output["results"]["lowestEnergy"] = sample_df.iloc[
            lowestEnergyIndex]["isingCost"]
        self.output["results"]["lowestEnergyProcessedFlow"] = sample_df.iloc[
            lowestEnergyIndex]["optimizedCost"]
        self.output["results"]["closestPowerProcessedFlow"] = sample_df.iloc[
            closestTotalPowerIndex]["optimizedCost"]
        self.output["results"]["bestProcessedFlow"] = sample_df[
            "optimizedCost"].min()
        self.output["results"]["postprocessingTime"] = time.perf_counter() \
                                                       - tic

    def solveFlowproblem(self, graph, generatorState):
        """
        solves the flow problem given in graph that corresponds to the
        generatorState in network. Calculates cost for a kirchhoffFactor
        of 1 and writes it to results["results"] under costKey. If
        costKey is None, it runs silently and only returns the computed
        cost value. The generatorState is fixed, so if a costKey is
        given, the function only returns a dictionary of the values it
        computes for an optimal flow. The cost is written to the field
        in results["results"]. Flow solutions don't spread imbalances
        across all buses so they can still be improved.
        """
        FlowSolution = edmonds_karp(graph, "superSource", "superSink")
        # key errors occur iff there is no power generated or no load at a bus.
        # Power can still flow through the bus, but no cost is incurred
        totalCost = 0
        for bus in self.network.buses.index:
            try:
                totalCost += (FlowSolution['superSource'][bus]['capacity']
                              - FlowSolution['superSource'][bus]['flow']) ** 2
            except KeyError:
                pass
            try:
                totalCost += (FlowSolution[bus]['superSink']['capacity']
                              - FlowSolution[bus]['superSink']['flow']) ** 2
            except KeyError:
                pass
        return totalCost

    # quantum computation struggles with finetuning powerflow to match
    # demand exactly. Using a classical approach to tune power flow can
    # archieved in polynomial time
    # TODO refactor this method
    def buildFlowproblem(self, generatorState, lineValues=None):
        """
        Build a self.networkx model to further optimise power flow.
        If using a warmstart, it uses the solution of the quantum
        computer encoded in generatorState to initialize a residual
        self.network. If the intial solution is good, a warmstart can
        speed up flow optimization by about 30%, but if it was bad, a
        warmstart makes it slower. warmstart is used if lineValues is
        not None.

        Args:
            generatorState: (list)
                List of all qubits with spin -1 in the solution.
            lineValues: (dict)
                Dictionary containing inital values for power flow.
        Returns:
            (networkx.DiGraph)
                The networkx formulation of the power flow problem.
        """
        # turn pypsa self.network in nx.DiGraph. Power generation and
        # consumption is modeled by adjusting capacity of the edge to a super
        # source/super sink
        graph = nx.DiGraph()
        graph.add_nodes_from(self.network.buses.index)
        graph.add_nodes_from(["superSource", "superSink"])

        for line in self.network.lines.index:
            bus0 = self.network.lines.loc[line].bus0
            bus1 = self.network.lines.loc[line].bus1
            cap = self.network.lines.loc[line].s_nom
            # if self.network has multiple lines between buses, make sure not
            # to erase the capacity of previous lines
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
                capacity=self.isingBackbone.calcTotalPowerGeneratedAtBus(
                    bus, generatorState)
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
                generatedPower = \
                    self.isingBackbone.calcTotalPowerGeneratedAtBus(
                        bus, generatorState)
                loadName = self.network.loads.index[
                    self.network.loads.bus == bus][0]
                load = self.network.loads_t['p_set'].iloc[0][loadName]
                netFlowThroughBus = 0
                for line in self.network.lines.index[self.network.lines.bus0
                                                     == bus]:
                    netFlowThroughBus += lineValues[(line, 0)]
                for line in self.network.lines.index[self.network.lines.bus1
                                                     == bus]:
                    netFlowThroughBus -= lineValues[(line, 0)]
                netPower = generatedPower - load + netFlowThroughBus

                graph["superSource"][bus]['flow'] = min(generatedPower,
                                                        generatedPower
                                                        - netPower)
                graph[bus]["superSink"]['flow'] = min(load, load + netPower)
        return graph

    def choose_sample(self, **kwargs):
        """
        After sampling a QUBO this chooses one sample to be returned as
        the solution.
        The sampleset has to be processed before to add additional
        information that will be used by a strategy of picking sample.
        So far, two strategies are supported 'LowestEnergy' returns the
        sample with the lowest energy state 'ClosestSample' returns the
        sample with the closes total power output to the total load.

        Returns:
            (pandas.Series)
                The chosen row to be used as the solution.
        """
        sample_df = self.getSampleDataframe()
        if kwargs['strategy'] == 'LowestEnergy':
            return sample_df.loc[sample_df['energy'].idxmin()]
        if kwargs['strategy'] == 'ClosestSample':
            closestSamples = sample_df[
                sample_df['deviation_from_opt_load'] == sample_df[
                    'deviation_from_opt_load'].min()
                ]
            return closestSamples.loc[closestSamples['optimizedCost'].idxmin()]

    def saveSample(self, sampleset):
        """
        Saves the sampleset as a data frame to be used by other methods
        and in the output dictionary since it contains all solutions
        found by the solver. Also writes additional information about
        the quantum annealing run into the output.
    
        Args:
            sampleset: (SampleSet)
                Record of all samples and additional data on them.
        Returns:
            (None)
                Modifies `self.sample_df` and `self.output`.
        """
        super().saveSample(sampleset)
        self.output["results"]["optimizationTime"] = \
        self.output["results"]["serial"]["info"]["timing"][
            "qpu_access_time"] / (10.0 ** 6)
        self.output["results"]["annealReadRatio"] = float(
            self.config["BackendConfig"]["annealing_time"]) / float(
            self.config["BackendConfig"]["num_reads"])
        self.output["results"]["totalAnnealTime"] = float(
            self.config["BackendConfig"]["annealing_time"]) * float(
            self.config["BackendConfig"]["num_reads"])
        # intentionally round totalAnnealTime so computations with similar
        # anneal time can ge grouped together
        self.output["results"]["mangledTotalAnnealTime"] = int(
            self.output["results"]["totalAnnealTime"] / 1000.0)


class DwaveReadQPU(DwaveCloudDirectQPU):
    """
    This class behaves like it's parent except it doesn't
    use the Cloud. Instead it reads a serialized Sample and pretends
    that it got that from the cloud
    """

    def getSampler(self):
        """
        This function returs nothing, but sets the path to the file that 
        contains the sample data to be returned when a sample request is
        made.

        Returns:
            (None)
                Modifies `self.inputFilePath` and `self.sampler`.
        """
        self.inputFilePath = "/energy/results_qpu/" + \
                             self.config["BackendConfig"]["sampleOrigin"]
        self.sampler = self.inputFilePath
        return None

    def getSampleSet(self):
        """
        Returns the sample set saved in the file which path is stored
        in `self.sampler`.
    
        Returns:
            (SampleSet)
                The d-wave sample read from the given filepath.
        """
        print(f"reading from {self.inputFilePath}")
        with open(self.inputFilePath) as inputFile:
            self.inputData = json.load(inputFile)
        return dimod.SampleSet.from_serializable(self.inputData["serial"])
