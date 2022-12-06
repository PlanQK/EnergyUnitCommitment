"""This module is for providing optimizers using d-wave's cloud for access
to their quantum hardware. So far, this module supports:
- quantum annealing (using pegasus topology)
- a hybrid solver
- tabu search (ocean package, classical algorithm)
- steepest descent (ocean package, classical algorithm)

For local testing and in case the processing of samples changes, this
module can also run quantum annealing by reading samples and returning
them as the result of a run"""

import abc

import time

# import packages for reading files
from os import path
from glob import glob
import json

import pandas
import dimod
import greedy

# import packages for flow optimization
import networkx as nx
from networkx.algorithms.flow import edmonds_karp

# importing d-wave packages
from dwave.system import LeapHybridSampler, DWaveSampler
from dwave.system import FixedEmbeddingComposite, EmbeddingComposite
from tabu import TabuSampler

# importing local QUBO modelling packages
from .backend_base import BackendBase
from .input_reader import InputReader

try:
    from libs.qubo_transformator import QuboTransformator
    from libs.qubo_transformator.ising_backbone import IsingBackbone
except ImportError:
    from ..qubo_transformator import QuboTransformator
    from ..qubo_transformator.ising_backbone import IsingBackbone


class AbstractDwaveSampler(BackendBase):
    """
    The base class for optimizers using d-wave's solvers. the optimization
    is done by a sampler, which is configured  in the child classes
    """

    def __init__(self, reader: InputReader):
        """
        Constructor for the D-WaveTabuSampler. It requires an
        InputReader, which handles the loading of the network and
        configuration file and a method for obtaining the correct
        sampler.

        Args:
            reader: (InputReader)
                 Instance of an InputReader, which handled the loading
                 of the network and configuration file.
        """
        super().__init__(reader=reader)
        self.sample_df = None
        self.ising_backbone = None
        self.sampler = None
        self.get_sampler()

    @abc.abstractmethod
    def get_sampler(self) -> dimod.Sampler:
        """
        Returns the D-Wave sampler and stores it as the attribute sampler.
        This method will be overridden in subclasses by choosing different
        samplers.

        Returns:
            (dimod.Sampler)
                The optimizer that generates samples of solutions.
        """
        raise NotImplementedError

    def process_samples(self, sampleset: dimod.SampleSet) -> pandas:
        """
        processes a returned sample set and constructs a pandas
        dataframe with all available information of that set.
        If necessary, this method will also add additional columns
        containing derived values.

        Args:
            sampleset: (dimod.SampleSet)
                The sampleset that is returned by the D-Wave solver.
        Returns:
            (pandas)
                A DataFrame containing all relevant data of the samples.
        """
        processed_samples_df = sampleset.to_pandas_dataframe()
        processed_samples_df["ising_cost"] = processed_samples_df.apply(
            lambda row: self.ising_backbone.calc_cost(
                [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1,
        )
        processed_samples_df["marginal_cost"] = processed_samples_df.apply(
            lambda row: self.ising_backbone.calc_marginal_cost(
                [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1,
        )
        processed_samples_df["total_power"] = processed_samples_df.apply(
            lambda row: self.ising_backbone.calc_total_power_generated(
                [idx for idx in range(len(row)) if row.iloc[idx] == -1]
            ),
            axis=1,
        )
        return processed_samples_df

    def process_solution(self) -> None:
        """
        Gets and writes info about the sample_df containing all samples and
        writes it in the `self.output` dictionary.

        Returns:
            (None)
                Modifies self.output.
        """
        best_sample = self.choose_sample()
        result_info = self.ising_backbone.generate_report(
            [qubit for qubit, qubit_spin in best_sample.items() if qubit_spin == -1]
        )
        self.output["results"] = {**self.output["results"], **result_info}

    def transform_problem_for_optimizer(self) -> None:
        """
        Initializes an ising_interface-instance, which encodes the Ising
        Spin Glass Problem, using the network to be optimized.

        Returns:
            (IsingBackbone)
                The ising_interface-instance, which encodes the Ising
                Spin Glass Problem.
        """
        print("transforming Problem...")
        self.ising_backbone = QuboTransformator.transform_network_to_qubo(
            self.network, self.config["ising_interface"]
        )

    def get_dimod_model(self) -> dimod.BinaryQuadraticModel:
        """
        Returns the corresponding D-Wave dimod.BinaryQuadraticModel to the
        model stored in self.ising_backbone as an IsingBackbone object

        Returns:
            (dimod.BinaryQuadraticModel)
                The equivalent D-Wave model of the IsingBackbone.

        """
        # store the directional qubits first, then the line's binary
        # representations
        linear = {
            spins[0]: strength
            for spins, strength in self.ising_backbone.get_ising_coefficients().items()
            if len(spins) == 1
        }
        # the convention is different to the sqa solver:
        # need to add a minus to the couplings
        quadratic = {
            spins: -strength
            for spins, strength in self.ising_backbone.get_ising_coefficients().items()
            if len(spins) == 2
        }
        return dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.Vartype.SPIN)

    def get_sample_dataframe(self) -> pandas.DataFrame:
        """
        Returns the data frame containing the data of the samples and
        their derived data.

        Returns:
            (pandas.DataFrame)
                The data frame containing a sample in each row.
        """
        return self.sample_df

    def choose_sample(self) -> pandas.Series:
        """
        After sampling a QUBO this function chooses one sample to be
        returned as the solution.
        Because the sampler of this class only returns one sample,
        there is only one sample to choose from. This function is
        overwritten in subclasses that return more than one sample.

        Returns:
            (pandas.Series)
                The chosen row to be used as the solution
        """
        return self.get_sample_dataframe().iloc[0]

    #
    def transform_solution_to_network(self) -> None:
        """
        Encodes the optimal solution found during optimization in a
        pypsa.Network and stores it in self.output. It reads the
        solution stored in the optimizer instance, prints some
        information about it and then writes it to the network.

        Returns:
            (None)
                Stores the output_network as dictionary in self.output.
        """
        best_sample = self.choose_sample()

        output_network = self.ising_backbone.set_output_network(
            solution=[
                qubit for qubit, qubit_spin in best_sample.items() if qubit_spin == -1
            ]
        )
        output_dataset = output_network.export_to_netcdf()
        self.output["network"] = output_dataset.to_dict()

    def optimize(self) -> None:
        """
        Optimizes the problem encoded in the IsingBackbone-Instance
        using Tabu search.

        Returns:
            (None)
                The optimized solution is stored in self.output.
        """
        print("starting optimization...")
        sampleset = self.get_sample_set()
        self.sample_df = self.process_samples(sampleset)
        self.save_sample(sampleset)
        print("done")

    def get_sample_set(self) -> dimod.SampleSet:
        """
        Queries the sampler to sample the problem and return all
        solutions.
        Overwriting this method allows alternative ways to get samples
        that contain solutions. This is especially useful for reading
        serialized samples from disk.

        Returns:
            (dimod.SampleSet)
                A record of the samples and any data returned by the
                sampler.

        """
        return self.sampler.sample(self.get_dimod_model())

    def save_sample(self, sampleset: dimod.SampleSet) -> None:
        """
        Saves the sampleset as a data frame to be used by other methods
        and in the output dictionary since it contains all solutions
        found by the solver.

        Args:
            sampleset: (dimod.SampleSet)
                Record of all samples and additional data on them.
        Returns:
            (None)
                Modifies `self.sample_df` and `self.output`.
        """
        if not hasattr(self, "sample_df"):
            self.sample_df = sampleset.to_pandas_dataframe()
        self.output["results"]["sample_df"] = self.sample_df.to_dict("split")
        self.output["results"]["serial"] = sampleset.to_serializable()

    def check_input_size(self, limit: float = 60.0):
        """
        checks if the estimated runtime is longer than the given limit

        Args:
            limit: an integer that is a measure for how long the limit is.
                    This is not a limit in seconds

        Returns: Doesn't return anything but raises an Error if it would take
                to long
        """
        if self.ising_backbone.num_interactions() >= 10000 * limit:
            raise ValueError("the estimated runtime is too long")


class DwaveTabu(AbstractDwaveSampler):
    """
    An optimizer using Tabu search to optimize a QUBO
    """

    def get_sampler(self) -> dimod.Sampler:
        """
        Sets the sampler using Tabu searchs to solve QUBO's

        Returns:
            (dimod.Sampler)
                The optimizer that generates samples of solutions.
        """
        self.sampler = TabuSampler()
        return self.sampler


class DwaveSteepestDescent(AbstractDwaveSampler):
    """
    An optimizer using steepest descent to solve a QUBO
    """

    def get_sampler(self) -> dimod.Sampler:
        """
        Sets the sampler using Tabu search to solve QUBO's

        Returns:
            (dimod.Sampler)
                The optimizer that generates samples of solutions.
        """
        self.sampler = greedy.SteepestDescentSolver()
        return self.sampler


class DwaveCloudSampler(AbstractDwaveSampler):
    """
    Class for structuring the class hierarchy. Inherits from AbstractDwaveSampler.
    Subclasses of sampler that use d-wave cloud services inherit from this
    class
    """

    pass


class DwaveCloudHybrid(DwaveCloudSampler):
    """
    Class inheriting from DwaveCloudSampler. It will use a hybrid solver to
    solve the given Ising spin glass problem.
    """

    def get_sampler(self) -> None:
        """
        Returns a D-Wave sampler that will query the hybrid solver for
        solving the Ising problem.
        In order to appropriately configure the solver, this method will
        also read the config attribute and save some settings as
        attributes.

        Returns:
            (None)
                Initializes various attributes.
        """
        self.token = self.config["API_token"]["dwave_API_token"]
        self.solver = "hybrid_binary_quadratic_model_version2"
        self.sampler = LeapHybridSampler(solver=self.solver, token=self.token)
        self.output["results"]["solver_id"] = self.solver

    def get_sample_set(self) -> dimod.SampleSet:
        """
        A method for obtaining a solution from d-wave using their hybrid
        solver.
        Since this queries the d-wave servers, it will have to wait for
        an answer which might take a while. Sampling will not abort if
        the response takes too long.

        Returns:
            (dimod.SampleSet)
                The sampleset containing a solution.
        """
        sampleset = super().get_sample_set()
        print("Waiting for server response...")
        # wait for response, no safeguard for endless looping
        while True:
            if sampleset.done():
                break
            time.sleep(2)
        return sampleset


class DwaveCloudDirectQPU(DwaveCloudSampler):
    """
    Class inheriting from DwaveCloudSampler. It will try to solve the given
    Ising spin glass problem on the D-Wave's cloud based QPU.
    """

    def __init__(self, reader: InputReader):
        """
        Constructor for the DwaveCloudDirectQPU. It requires an
        InputReader, which handles the loading of the network and
        configuration file. In addition, it is made sure, that the
        timeout is set correctly in self.config.

        Args:
            reader: (InputReader)
                 Instance of an InputReader, which handled the loading
                 of the network and configuration file.
        """
        super().__init__(reader=reader)
        if self.config["backend_config"]["timeout"] < 0:
            self.config["backend_config"]["timeout"] = 3600

    @staticmethod
    def get_filepaths(root_path: str, file_regex: str) -> list[str]:
        """
        Returns the filepath composed of 'root_path' and 'file_regex'.

        Args:
            root_path: (str)
                The root path.
            file_regex: (str)
                An addition to add to 'root_path'.

        Returns:
            (str)
                The combined filepath.
        """
        return glob(path.join(root_path, file_regex))

    def check_input_size(self, limit: float = 60.0):
        """
        this sets a limit on the heuristic used for embedding the QUBO onto
        the working graph of the annealer.

        Args:
            limit: a float that is a measure for how long we can try to find
                   an embedding

        Returns: modifies the timeout parameter, which limits how long
                 the heuristic can search for an embedding
        """
        self.config["backend_config"]["timeout"] = min(
            self.config["backend_config"].get("timeout", 3600), limit
        )

    def get_sampler(self) -> dimod.Sampler:
        """
        Returns a D-Wave sampler that will query the quantum annealer
        (pegasus topology) for solving the ising problem.
        In order to appropriately configure the solver, this method will
        also read the config attribute and save some settings as
        attributes. If a fitting embedding is found, it will be reused
        to embed the theoretical QUBO model onto the hardware.

        Returns:
            (dimod.Sampler)
                The optimizer that generates samples of quantum
                annealing runs.
        """
        self.token = self.config["API_token"]["dwave_API_token"]
        # pegasus topology corresponds to Advantage 4.1
        direct_sampler = DWaveSampler(
            solver={"qpu": True, "topology__type": "pegasus"}, token=self.token
        )
        if hasattr(self, "embedding"):
            self.sampler = FixedEmbeddingComposite(direct_sampler, self.embedding)
        else:
            self.sampler = EmbeddingComposite(direct_sampler)
        return self.sampler

    def get_sample_set(self) -> dimod.SampleSet:
        """
        Queries the quantum annealer to sample the problem and return
        all solutions.
        Parameters for configuring the quantum annealer are read from
        the `config` attribute.

        Returns:
            (dimod.SampleSet)
                A record of the quantum annealing samples returned by
                the sampler.
        """
        # construct annealer arguments. necessary to differentiate if you are
        # using a preexisting embedding or have to find one for this run
        sample_arguments = {
            arg: val
            for arg, val in self.config["backend_config"].items()
            if arg
            in [
                "num_reads",
                "annealing_time",
                "chain_strength",
                "programming_thermalization",
                "readout_thermalization",
            ]
        }
        if not hasattr(self, "embedding"):
            sample_arguments["embedding_parameters"] = dict(
                timeout=self.config["backend_config"]["timeout"]
            )
            sample_arguments["return_embedding"] = True
        try:
            sampleset = self.sampler.sample(**sample_arguments)
        except ValueError:
            print("no embedding found in given time limit")
            raise ValueError("no embedding onto qpu was found") from None
        print("Waiting for server response...")
        while True:
            if sampleset.done():
                break
            time.sleep(1)
        return sampleset

    def optimize_sample_flow(self, sample: pandas.Series) -> int:
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
        generator_state = [
            qubit for qubit, qubit_spin in sample.items() if qubit_spin == -1
        ]
        graph = self.build_flow_problem(generator_state)
        return self.solve_flow_problem(
            graph,
        )

    def process_samples(self, sampleset: dimod.SampleSet) -> pandas.DataFrame:
        """
        A method for turning a sampleset into a data frame and add
        additional data.
        The method transforms the sampleset into the corresponding data
        frame. Then this class and subclasses overriding this method use
        pandas to add new columns containing additional information
        about each sample.

        Args:
            sampleset: (dimod.SampleSet)
                A d-wave sampleset containing the annealing data.
        Returns:
            (pandas.DataFrame)
                The data frame of the sample set with extra columns.
        """
        processed_samples_df = super().process_samples(sampleset)
        processed_samples_df["optimized_cost"] = processed_samples_df.apply(
            lambda row: self.optimize_sample_flow(
                row[:-3],
            ),
            axis=1,
        )
        total_load = 0.0
        for idx, _ in enumerate(self.network.snapshots):
            total_load += self.ising_backbone.get_total_load(idx)
        processed_samples_df["deviation_from_opt_load"] = processed_samples_df.apply(
            lambda row: abs(
                total_load
                - self.ising_backbone.calc_total_power_generated(
                    [qubit for qubit, qubit_spin in row.items() if qubit_spin == -1]
                )
            ),
            axis=1,
        )
        return processed_samples_df

    def process_solution(self):
        """
        Gets and writes info about the sample_df and writes it in the
        `self.output` dictionary. Inherits from its parent class and sets
        additional values in self.output.

        Returns:
            (None)
                Modifies self.output.
        """
        tic = time.perf_counter()
        super().process_solution()
        lowest_energy_index = self.sample_df["energy"].idxmin()
        self.output["results"]["lowest_energy"] = self.sample_df.iloc[
            lowest_energy_index
        ]["ising_cost"]
        closest_samples = self.sample_df[
            self.sample_df["deviation_from_opt_load"]
            == self.sample_df["deviation_from_opt_load"].min()
        ]
        closest_total_power_index = closest_samples["energy"].idxmin()
        self.output["samples_df"] = self.sample_df.to_dict("index")
        self.output["results"]["lowest_energy"] = self.sample_df.iloc[
            lowest_energy_index
        ]["ising_cost"]
        self.output["results"]["lowest_energy_processed_flow"] = self.sample_df.iloc[
            lowest_energy_index
        ]["optimized_cost"]
        self.output["results"]["closest_power_processed_flow"] = self.sample_df.iloc[
            closest_total_power_index
        ]["optimized_cost"]
        self.output["results"]["best_processed_flow"] = self.sample_df[
            "optimized_cost"
        ].min()
        self.output["results"]["postprocessing_time"] = time.perf_counter() - tic

    def solve_flow_problem(self, graph) -> int:
        """
        solves the flow problem given in graph that corresponds to the
        generator_state in network. Calculates cost for a kirchhoffFactor
        of 1 and writes it to results["results"] under costKey. If
        costKey is None, it runs silently and only returns the computed
        cost value. The generator_state is fixed, so if a costKey is
        given, the function only returns a dictionary of the values it
        computes for an optimal flow. The cost is written to the field
        in results["results"]. Flow solutions don't spread imbalances
        across all buses, so they can still be improved.
        """
        flow_solution = edmonds_karp(graph, "super_source", "super_sink")
        # key errors occur iff there is no power generated or no load at a bus.
        # Power can still flow through the bus, but no cost is incurred
        total_cost = 0
        for bus in self.network.buses.index:
            try:
                total_cost += (
                    flow_solution["super_source"][bus]["capacity"]
                    - flow_solution["super_source"][bus]["flow"]
                ) ** 2
            except KeyError:
                pass
            try:
                total_cost += (
                    flow_solution[bus]["super_sink"]["capacity"]
                    - flow_solution[bus]["super_sink"]["flow"]
                ) ** 2
            except KeyError:
                pass
        return total_cost

    # quantum computation struggles with fine tuning powerflow to match
    # demand exactly. Using a classical approach to tune power flow can
    # achieved in polynomial time
    def build_flow_problem(
        self, generator_state: list, line_values: dict = None
    ) -> nx.DiGraph:
        """
        Build a self.networkx model to further optimise power flow.
        If using a warmstart, it uses the solution of the quantum
        computer encoded in generatorState to initialize a residual
        self.network. If the initial solution is good, a warmstart can
        speed up flow optimization by about 30%, but if it was bad, a
        warmstart makes it slower. warmstart is used if lineValues is
        not None. This only optimizes the first snapshot of the network

        Args:
            generator_state: (list)
                List of all qubits with spin -1 in the solution.
            line_values: (dict)
                Dictionary containing initial values for power flow.
        Returns:
            (networkx.DiGraph)
                The networkx formulation of the power flow problem.
        """
        # turn pypsa self.network in nx.DiGraph. Power generation and
        # consumption is modeled by adjusting capacity of the edge to a super
        # source/super sink
        graph = nx.DiGraph()
        graph.add_nodes_from(self.network.buses.index)
        graph.add_nodes_from(["super_source", "super_sink"])

        for line in self.network.lines.index:
            bus0 = self.network.lines.loc[line].bus0
            bus1 = self.network.lines.loc[line].bus1
            cap = self.network.lines.loc[line].s_nom
            # if self.network has multiple lines between buses, make sure not
            # to erase the capacity of previous lines
            if graph.has_edge(bus0, bus1):
                graph[bus0][bus1]["capacity"] += cap
                graph[bus1][bus0]["capacity"] += cap
            else:
                graph.add_edges_from(
                    [(bus0, bus1, {"capacity": cap}), (bus1, bus0, {"capacity": cap})]
                )

        for bus in self.network.buses.index:
            graph.add_edge(
                "super_source",
                bus,
                capacity=self.ising_backbone.calc_total_power_generated_at_bus(
                    bus, generator_state, time=0
                ),
            )
        for load in self.network.loads.index:
            graph.add_edge(
                self.network.loads.loc[load].bus,
                "super_sink",
                capacity=self.network.loads_t["p_set"].iloc[0][load],
            )
        # done building nx.DiGraph

        if line_values is not None:
            # generate flow for self.network lines
            for line in self.network.lines.index:
                bus0 = self.network.lines.loc[line].bus0
                bus1 = self.network.lines.loc[line].bus1
                if hasattr(graph[bus1][bus0], "flow"):
                    graph[bus1][bus0]["flow"] -= line_values[(line, 0)]
                else:
                    graph[bus0][bus1]["flow"] = line_values[(line, 0)]

            # adjust source and sink flow to make it a valid flow. edges
            # to source/sink are not at full capacity iff there is a net
            # demand/power generated after subtraction power flow at that bus
            # might be wrong if kirchhoff constraint was violated in quantum
            # solution
            for bus in self.network.buses.index:
                generated_power = self.ising_backbone.calc_total_power_generated_at_bus(
                    bus, generator_state
                )
                load_name = self.network.loads.index[self.network.loads.bus == bus][0]
                load = self.network.loads_t["p_set"].iloc[0][load_name]
                net_flow_through_bus = 0
                for line in self.network.lines.index[self.network.lines.bus0 == bus]:
                    net_flow_through_bus += line_values[(line, 0)]
                for line in self.network.lines.index[self.network.lines.bus1 == bus]:
                    net_flow_through_bus -= line_values[(line, 0)]
                net_power = generated_power - load + net_flow_through_bus

                graph["super_source"][bus]["flow"] = min(
                    generated_power, generated_power - net_power
                )
                graph[bus]["super_sink"]["flow"] = min(load, load + net_power)
        return graph

    def choose_sample(self, **kwargs) -> pandas.Series:
        """
        After sampling a QUBO this chooses one sample to be returned as
        the solution.
        The sampleset has to be processed before to add additional
        information that will be used by a strategy of picking sample.
        So far, two strategies are supported 'lowest_energy' returns the
        sample with the lowest energy state 'closest_sample' returns the
        sample with the closes total power output to the total load.

        Returns:
            (pandas.Series)
                The chosen row to be used as the solution.
        """
        sample_df = self.get_sample_dataframe()
        strategy = kwargs.get("strategy", "lowest_energy")
        if strategy == "lowest_energy":
            return sample_df.loc[sample_df["energy"].idxmin()]
        elif strategy == "closest_sample":
            closest_samples = sample_df[
                sample_df["deviation_from_opt_load"]
                == sample_df["deviation_from_opt_load"].min()
            ]
            return closest_samples.loc[closest_samples["optimized_cost"].idxmin()]
        raise ValueError(f"The strategy {strategy} is not valid for choosing a sample")

    def save_sample(self, sampleset: dimod.SampleSet) -> None:
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
        super().save_sample(sampleset)
        self.output["results"]["optimization_time"] = self.output["results"]["serial"][
            "info"
        ]["timing"]["qpu_access_time"] / (10.0**6)
        self.output["results"]["anneal_read_ratio"] = float(
            self.config["backend_config"]["annealing_time"]
        ) / float(self.config["backend_config"]["num_reads"])
        self.output["results"]["total_anneal_time"] = float(
            self.config["backend_config"]["annealing_time"]
        ) * float(self.config["backend_config"]["num_reads"])
        # intentionally round total_anneal_time so computations with similar
        # anneal time can ge grouped together
        self.output["results"]["mangled_total_anneal_time"] = int(
            self.output["results"]["total_anneal_time"] / 1000.0
        )


class DwaveReadQPU(DwaveCloudDirectQPU):
    """
    This class behaves like it's parent except it doesn't
    use the Cloud. Instead, it reads a serialized Sample and pretends
    that it got that from the cloud.
    """

    def get_sampler(self) -> None:
        """
        This function returns nothing, but sets the path to the file that
        contains the sample data to be returned when a sample request is
        made.

        Returns:
            (None)
                Modifies `self.input_file_path` and `self.sampler`.
        """
        self.input_file_path = (
            "/energy/results_qpu/" + self.config["backend_config"]["sample_origin"]
        )
        self.sampler = self.input_file_path

    def get_sample_set(self) -> dimod.SampleSet:
        """
        Returns the sample set saved in the file which path is stored
        in `self.sampler`.

        Returns:
            (dimod.SampleSet)
                The d-wave sample read from the given filepath.
        """
        print(f"reading from {self.input_file_path}")
        with open(self.input_file_path, encoding="utf-8") as input_file:
            input_data = json.load(input_file)
        return dimod.SampleSet.from_serializable(input_data["serial"])
