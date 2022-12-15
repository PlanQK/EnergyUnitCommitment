"""This module serves as a wrapper for the simulated quantum annealing solver.
You can find more information on the implementation of it at
https://doi.org/10.5905/ethz-1007-127
It also provides a solver implementing classical annealing by adjusting the
transverse field to be 0 everywhere
"""

import random
import time
from ast import literal_eval

from numpy import sqrt

from .input_reader import InputReader, GraphReader
from .backend_base import BackendBase

# try import from local .so
# Error message for image: herrd1/siquan:latest
# /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
# (required by /energy/libs/backends/siquan.cpython-39-x86_64-linux-gnu.so)`
try:
    from . import siquan
    from ..qubo_transformator import QuboTransformator, TspTransformator
# try import from installed module siquan
except ImportError:
    from libs.qubo_transformator import QuboTransformator, TspTransformator
    import siquan


class ClassicalBackend(BackendBase):
    """
    A class for solving the unit commitment problem using classical
    annealing. This is done using the SiQuAn solver and setting the
    transverse Field to zero.
    """

    def __init__(self, reader: InputReader):
        """
        Constructor for the ClassicalBackend class. It requires an
        InputReader, which handles the loading of the network and
        configuration file.

        Args:
            reader: (InputReader)
                 Instance of an InputReader, which handled the loading
                 of the network and configuration file.
        """
        super().__init__(reader=reader)
        self.siquan_config = self.config.setdefault("backend_config", {})
        self.siquan_config.setdefault("seed", random.randrange(10**6))
        self.siquan_config["transverse_field_schedule"] = self.get_h_schedule()
        self.siquan_config.setdefault("temperature_schedule", "[0.1,iF,0.0001]")
        self.siquan_config.setdefault("trotter_slices", 32)
        self.siquan_config.setdefault("optimization_cycles", 16)
        self.solver = siquan.DTSQA()
        self.configure_solver()

    def transform_problem_for_optimizer(self) -> None:
        """
        Initializes an IsingInterface-instance, which encodes the Ising
        Spin Glass Problem, using the network to be optimized.

        Returns:
            (None)
                Add the IsingInterface-instance to
                self.transformed_problem.
        """
        print("transforming problem...")
        self.transformed_problem = QuboTransformator.transform_network_to_qubo(
            self.network, self.config["ising_interface"]
        )

    def transform_solution_to_network(self) -> None:
        """
        Encodes the optimal solution found during optimization and
        stored in `self.output` into a pypsa.Network. It reads the
        solution stored in the optimizer instance, prints some
        information regarding it to stdout and then writes it into a
        network, which is then saved in self.output.

        Returns:
            (None)
                Modifies `self.output` with the output_network.
        """
        output_network = self.transformed_problem.set_output_network(
            solution=self.output["results"]["state"]
        )
        output_dataset = output_network.export_to_netcdf()
        self.output["network"] = output_dataset.to_dict()

    def optimize(self) -> None:
        """
        Optimizes the problem encoded in the IsingBackbone-Instance.
        It uses the siquan solver which parameters can be set using
        self.config. Configuration of the solver is delegated to a
        method that can be overwritten in child classes.

        Returns:
            (None)
                The optimized solution is stored in the `self.output`
                dictionary.
        """
        print("starting optimization...")
        tic = time.perf_counter()
        result = self.solver.minimize(
            self.transformed_problem.siquan_format(),
            self.transformed_problem.num_variables(),
        )
        self.output["results"]["optimization_time"] = time.perf_counter() - tic
        # parse the entry in "state" before using it
        result["state"] = literal_eval(result["state"])
        self.write_results_to_output(result)
        print("done")

    def get_h_schedule(self) -> str:
        """
        A method for getting the transverse field schedule in the
        configuration. For classical annealing, there is no transverse
        field, so it always returns the same config string. Overwriting
        this method can be used to get non-zero transverse field.

        Returns:
            (str)
                The configuration string for the transverse field of the
                siquan solver for classical annealing.
        """
        return "[0]"

    def configure_solver(self) -> None:
        """
        Reads and sets siquan solver parameter from the config dict.
        Solver configuration is read from the config dict and default
        values are used if it is not specified in the configuration.
        These default values are not suitable for solving large
        problems. Since classical annealing and simulated quantum
        annealing only differs in the transverse field, setting that
        field is in its own method, so it can be overwritten.

        Returns:
            (None)
                Modifies `self.solver` and sets hyperparameters
        """
        self.solver.setSeed(self.siquan_config["seed"])
        self.solver.setHSchedule(self.siquan_config["transverse_field_schedule"])
        self.solver.setTSchedule(self.siquan_config["temperature_schedule"])
        self.solver.setTrotterSlices(int(self.siquan_config["trotter_slices"]))
        self.solver.setSteps(int(self.siquan_config["optimization_cycles"]))

    def print_solver_specific_report(self) -> None:
        """
        Prints additional information about the solution that is solver specific.
        The only non-generic information is the energy of the solution regarding
        the Ising spin glass formulation.

        Returns:
            (None)
                Prints information to stdout.
        """
        print(
            f"Total energy cost of QUBO (with constant terms): "
            f"{self.output['results']['total_cost']}"
        )
        print("---")
        print(f"Sqa runtime: {self.output['results']['runtime_sec']}")
        print(f"Sqa runtime cycles: {self.output['results']['runtime_cycles']}")
        print(f"Ising Interactions: {self.transformed_problem.num_interactions()}")

    def write_results_to_output(self, result: dict) -> None:
        """
        This writes solution specific values of the optimizer result
        and the Ising spin glass problem solution the self.output.

        Args:
            result: (dict)
                The python dictionary returned from the sqa solver.

        Returns:
            (None)
                Modifies `self.output` with solution specific parameters
                and values.
        """
        for key in result:
            self.output["results"][key] = result[key]
        self.output["results"] = {
            **self.output["results"],
            **self.transformed_problem.generate_report(result["state"]),
        }

    def check_input_size(self, limit: float = 60.0) -> None:
        """
        Checks if the estimated runtime is longer than the given limit

        Args:
            limit: (float)
                An integer that is a measure for how big the limit is.
                This is not a limit in seconds because that depends on
                the hardware this is running on

        Returns: (None)
            Raises an Error if the calculation would take too long
        """
        runtime_factor = self.transformed_problem.num_interactions() * 0.001
        runtime_factor *= self.siquan_config["trotter_slices"] * 0.001
        runtime_factor *= self.siquan_config["optimization_cycles"] * 0.001
        used_limit = runtime_factor / limit
        if used_limit >= 1.0:
            raise ValueError("the estimated runtime is too long")


class SqaBackend(ClassicalBackend):
    """
    A class for solving the unit commitment problem using classical
    annealing. This is done using the SiQuAn solver and setting the
    transverse Field to a value given in the configuration file. It
    inherits from ClassicalBackend and only overwrites the method
    get_h_schedule.
    """

    def get_h_schedule(self) -> str:
        """
        A method for getting the transverse field schedule from the
        configuration and returns it.

        Returns:
            (str)
                The configuration of the 'transverseFieldSchedule', to
                set in siquan solver, according to self.config.
        """
        return self.config["backend_config"].setdefault(
            "transverse_field_schedule", "[8.0,0.0]"
        )


class SqaIterator(BackendBase):
    """
    A class for solving the unit commitment problem using iterative qubos solved
    by an annealer
    """

    def __init__(self, reader: InputReader):
        """
        Constructor for the ClassicalBackend class. It requires an
        InputReader, which handles the loading of the network and
        configuration file.

        Args:
            reader: (InputReader)
                 Instance of an InputReader, which handled the loading
                 of the network and configuration file.
        """
        super().__init__(reader=reader)
        self.siquan_config = self.config["backend_config"]
        self.solver = SqaBackend.create_optimizer(reader=reader)
        self.solver_marginal_config = self.solver.config["ising_interface"][
            "marginal_cost"
        ]
        self.solver.configure_solver()
        self.intermediary_results = {}
        self.current_step = 0
        self.iteration_step = 0
        self.max_iteration = 0
        self.current_estimation = 0
        self.iteration_results = []

    def transform_problem_for_optimizer(self) -> None:
        """
        Initializes an IsingInterface-instance, which encodes the Ising
        Spin Glass Problem, using the network to be optimized.

        Returns:
            (None)
                Add the IsingInterface-instance to
                self.transformed_problem.
        """
        print("transforming problem...")
        self.iteration_step = 0
        self.max_iteration = self.config["backend_config"]["max_iteration"]
        if self.max_iteration <= 0:
            raise ValueError(
                f"Number of iteration steps {self.max_iteration} is not positive"
            )
        self.current_estimation = self.solver_marginal_config["target_cost"]
        self.iteration_results = []

    def calculate_step(self, solution_cost: float) -> float:
        """
        For the current ising problem and the calculated solution, this calculates the step size
        of the next estimation of the marginal costs to construct the next ising problem

        Args:
            solution_cost: (float)
                The calculated cost QUBO cost of the solution of the last iteration step

        Returns: (float)
            The step size for the next estimation of the marginal cost
        """
        return sqrt(solution_cost / self.solver_marginal_config["scale_factor"])

    def extract_from_sqa(self, sqa_result: dict) -> dict:
        """
        This extracts the relevant result data from an sqa run and strips
        information that is not relevant

        Args:
            sqa_result: (dict)
                The result of using the sqa solver to solve an intermediary QUBO

        Returns: (dict)
            A dictionary of the sqa run with fields that don't change over
            mutiple steps being ommitted.
        """
        result = sqa_result["results"]
        result["offset"] = sqa_result["config"]["ising_interface"]["marginal_cost"][
            "offset"
        ]
        result["target_cost"] = sqa_result["config"]["ising_interface"][
            "marginal_cost"
        ]["target_cost"]
        result["current_step"] = self.current_step
        return result

    def optimize(self) -> None:
        """
        Optimizes the problem encoded in the IsingBackbone-Instance.
        It uses the siquan solver which parameters can be set using
        self.config. Configuration of the solver is delegated to a
        method that can be overwritten in child classes.

        Returns:
            (None)
                The optimized solution is stored in the `self.output`
                dictionary.
        """
        tic = time.perf_counter()
        for step in range(self.max_iteration):
            print("\n ----------------------------------------")
            print(f"\n -- Step {step} -- Estimation: {self.current_estimation} ")
            print("\n ----------------------------------------")
            self.solver_marginal_config["target_cost"] = self.current_estimation
            self.solver.transform_problem_for_optimizer()
            self.solver.optimize()
            self.solver.print_report()
            current_result = self.solver.get_output()
            self.intermediary_results[self.current_step] = self.extract_from_sqa(
                current_result
            )
            self.current_step += 1
            self.current_estimation += self.calculate_step(
                current_result["results"]["total_cost"]
            )

            # Results that incurr kirchhoff costs below 1.0 count as "feasible"
            # in case of imprecisions due to float values
            if current_result["results"]["kirchhoff_cost"] < 1.0:
                self.output["results"]["termination_condition"] = "feasible"
                print(" \n--- Found feasible solution ---\n")
                break

        self.output["results"]["optimization_time"] = time.perf_counter() - tic
        if current_result["results"]["kirchhoff_cost"] >= 1.0:
            print("Didn't find a feasible solution in allotted number of iterations")
            self.output["results"]["termination_condition"] = "max_steps"
        current_result["intermediary_results"] = self.intermediary_results

        self.write_results_to_output(current_result)

    def write_results_to_output(self, result: dict) -> None:
        """
        This writes solution specific values of the optimizer result
        and used ising spin problems and their solutions to the `output`
        attribute

        Args:
            result: (dict)
                The python dictionary returned from the sqa solver.

        Returns:
            (None)
                Modifies `self.output` with solution specific parameters
                and values.
        """
        for key in result:
            if key == "results":
                continue
            self.output["results"][key] = result[key]
        self.output["results"] = {
            **self.output["results"],
            **self.solver.transformed_problem.generate_report(
                result["results"]["state"]
            ),
        }
