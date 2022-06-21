import math
from itertools import product

import numpy as np
import qiskit

try:
    from .IsingPypsaInterface import IsingBackbone  # import for Docker run
    from .BackendBase import BackendBase  # import for Docker run
except ImportError:
    from IsingPypsaInterface import IsingBackbone  # import for local/debug run
    from BackendBase import BackendBase  # import for local/debug run

from datetime import datetime
from numpy import median
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers import BaseBackend
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.algorithms.optimizers import SPSA, COBYLA, ADAM
from qiskit.circuit import Parameter, ParameterVector
from scipy import stats

from .InputReader import InputReader

try:
    from .IsingPypsaInterface import IsingBackbone  # import for Docker run
    from .BackendBase import BackendBase  # import for Docker run
except ImportError:
    from IsingPypsaInterface import IsingBackbone  # import for local/debug run
    from BackendBase import BackendBase  # import for local/debug run


class QaoaAngleSupervisor:
    """a class for choosing qaoa angles when making (multiple) runs in order to get
    well distributed samples. It does so by providing an iterator of parameter initialization.
    This iterator has a reference to the qaoa optimizer instance, so it can inspect it
    to update which parameter to get next and get the configuration info.
    """

    @classmethod
    def make_angle_supervisior(cls, qaoa_optimizer):
        """
        A factory method for returning the correct supervisior for the chosen strategy.

        The "RandomOrFixed" supervision will choose, based on a config list either a fixed
        float value or a random angle. After a set of repetitions, it will do so again
        but replace random initializations with the best angle initialization found so far

        The "GridSearch" will use a grid to evenly distribute initial parameters across the
        parameter space.
    
        Args:
            qaoa_optimizer: (QaoaQiskit) The qaoa optimizer that will consume the supplied angles
        Returns:
            (QaoaAngleSupervisor) An instance of subclass of a QaoaAngleSupervisor
        """
        supervisior_type = qaoa_optimizer.config_qaoa.get("supervisior_type", "RandomOrFixed")
        if supervisior_type == "RandomOrFixed":
            return QaoaAngleSupervisorRandomOrFixed(qaoa_optimizer)
        if supervisior_type == "GridSearch":
            return QaoaAngleSupervisorGridSearch(qaoa_optimizer)

    def get_initial_angle_iterator(self):
        """
        This returns an iterator for initial angle initialization. Iterating using this is the main
        way of using this class. By storing a reference to the executing qaoa optimizer, this class
        can adjust which parameters to use next based on qaoa results.
    
        Returns:
            (Iterator[np.array]) An iterator which yields initial angle values for the optimizer
        """
        raise NotImplementedError

    def get_num_angles(self):
        """This returns the number of angles  used for parametrization of the quantum circuit.
        This is necessary for correctly binding the constructed circuit to the angles"""
        raise NotImplementedError


class QaoaAngleSupervisorRandomOrFixed(QaoaAngleSupervisor):
    """a class for choosing initial qaoa angles. The strategy is given by a list. Either an angle parameter
    is fixed based on the list entry, or chosen randomly
    """

    def __init__(self, qaoa_optimizer):
        """
        Initializes all attributes necessary to build the iterator that the qaoa_optimizer can consume.
    
        Args:
            qaoa_optimizer: (QaoaQiskit) The qaoa optimizer that will consume the supplied angles
        Returns:
            (list) a list of float values to be used as angles in a qaoa circuit
        """
        self.qaoa_optimizer = qaoa_optimizer
        self.config_qaoa = qaoa_optimizer.config_qaoa
        self.problem_range = self.config_qaoa.get("problem_range", 2)
        self.mixing_range = self.config_qaoa.get("mixing_range", 1)
        self.config_guess = self.config_qaoa["initial_guess"]
        self.num_angles = len(self.config_guess)
        self.repetitions = self.config_qaoa["repetitions"]

    def get_num_angles(self):
        """This returns the number of angles are used for parametrization of the quantum circuit.
        This is necessary for correctly binding the constructed circuit to the angles"""
        return self.num_angles

    def get_best_initial_angles(self):
        """
        When calling this method, it searches the results of the stored qaoa_optimizer for the best result
        so far. Then it uses those values to construct an initial angle guess by substituting all angles
        that are set randomly according to the configuration with the best results.
    
        Returns:
            (np.array) a np array of values to be used as angles
        """
        min_cf_vars = self.qaoa_optimizer.get_min_cf_vars()
        self.qaoa_optimizer.output["results"]["initial_guesses"]["refined"] = min_cf_vars
        best_initial_guess = self.config_guess
        for j in range(self.num_angles):
            if best_initial_guess[j] == "rand":
                best_initial_guess[j] = min_cf_vars[j]
        return best_initial_guess

    def get_initial_angle_iterator(self):
        """
        returns an iterator that returns initial angle guesses to be consumed by the qaoa optimizer.
        These are constructed according to the self.config_guess list. If this list contains at least
        one random initialization, it will choose the best angle result so far and return those to be 
        used for more qaoa repetitions.
    
        Returns:
            (iterator[np.array]) description
        """
        for idx in range(self.repetitions):
            yield self.choose_initial_angles()

        if "rand" not in self.config_guess:
            return

        self.config_guess = self.get_best_initial_angles()
        for idx in range(self.repetitions):
            yield self.choose_initial_angles()

    def choose_initial_angles(self):
        """
        Method for returning a np.array of angles to be used for each layer in the qaoa circuit
    
        Returns:
            (np.array) a np.array of floats
        """
        initial_angles = []
        for idx, current_guess in enumerate(self.config_guess):
            # if chosen randomly, choose a random angle and scale based on whether it is the angle
            # of a problem hamiltonian sub circuit or mixing hamiltonian sub circuit
            if current_guess == "rand":
                next_angle = 2 * math.pi * (0.5 - np.random.rand())
                if idx % 2 == 0:
                    next_angle *= self.problem_range
                else:
                    next_angle *= self.mixing_range
            # The angle is fixed for all repetitions
            else:
                next_angle = current_guess
            initial_angles.append(next_angle)
        return np.array(initial_angles)


class QaoaAngleSupervisorGridSearch(QaoaAngleSupervisor):
    """a class for choosing qaoa parameters by searching using a grid on the given parameter space
    """

    def __init__(self, qaoa_optimizer):
        """
        first calculates the grid that is going to be searched and then sets up the data structures
        necessary to keep track which grid points have already been tried output

        A grid is a product of grids for each angle space of one layer of problem+mixing hamiltonian.
        We can give a default grid that each layer will use if their grid config doesn't specify it.
        A grid for one layers needs upper and lower bounds for the angle of the mixing and problem
        hamiltonian. Furthermore, it either needs to specify how many grid points are added for each angle
        Keep in mind that the total number of grids point is the product over all grids over all layers
        which can grow very quickly.
        Each Grid is represented as a dictionary with 6 Values
            lowerBoundMixing, upperBoundMixing, num_grid_mixing
            lower_bound_problem, upper_bound_problem, num_grid_problem
        If a value is not specified the value of a default grid is used instead
            default_grid: (dict) the default paramters of the grid for one layer
                keys
                    lower_bound_problem : -1
                    upper_bound_problem : 1
                    num_grid_problem : 3
                    lowerBoundMixing : -1
                    upperBoundMixing : 1
                    num_grid_mixing : 3
        The config file contains a list of all grids for the layers of the quantum circuit
            gridList: list
                a list of dictionaries. The i-th dictionary contains the grid values of the i-th layer
        """
        self.qaoa_optimizer = qaoa_optimizer
        self.config_qaoa = qaoa_optimizer.config_qaoa

        self.set_default_grid(self.config_qaoa.get('default_grid', {}))
        self.grids_by_layer = [grid for layer in self.config_qaoa['initial_guess']
                               for grid in self.transform_to_gridpoints(layer)]
        self.num_angles = len(self.grids_by_layer)

    def get_num_angles(self):
        """This returns the number of angles are used for parametrization of the quantum circuit.
        This is necessary for correctly binding the constructed circuit to the angles"""
        return self.num_angles

    def get_initial_angle_iterator(self):
        """
        returns an iterator that returns initial angle guesses to be consumed by the qaoa optimizer.
        Together, these initial angles form a grid on the angle space
   
        Returns:
            (Iterator[np.array]) An iterator which yields initial angle values for the optimizer
        """
        for angle_list in product(*self.grids_by_layer):
            yield np.array(angle_list)

    def set_default_grid(self, default_grid: dict):
        """
        reads a grid dictionary and saves them to be accessible later as a default fallback value 
        in case they aren't specified for one a layer of qaoa
    
        Args:
            default_grid: (dict)
                a dictionary with values to specify a grid. For this grid, there also
                exist default values to be used as default values
        Returns:
            (None) modifies the attribute self.default
        """
        self.default_grid = {
            "lower_bound_problem": default_grid.get("lower_bound_problem", - math.pi),
            "upper_bound_problem": default_grid.get("upper_bound_problem", math.pi),
            "num_gridpoints_problem": _grid.get("num_gridpoints_problem", 3),
            "lower_bound_mixing": default_grid.get("lower_bound_mixing", -math.pi),
            "upper_bound_mixing": default_grid.get("upper_bound_mixing", math.pi),
            "num_gridpoints_mixing": default_grid.get("num_gridpoints_mixing", 3),
        }

    def transform_to_gridpoints(self, grid_dict):
        """
        returns two list of grid points based on the dictionary describing the grid points of one layer.
    
        Args:
            grid_dict: (dict) a dicitonary with the following keys
                    lower_bound_problem 
                    upper_bound_problem 
                    num_grid_problem 
                    lowerBoundMixing 
                    upperBoundMixing 
                    num_grid_mixing 
        Returns:
            (list, list) returns two lists with float values
        """
        problem_grid = self.make_grid_list(
            lower_bound=grid_dict.get('lower_bound_problem', self.default_grid['lower_bound_problem']),
            upper_bound=grid_dict.get('upper_bound_problem', self.default_grid['upper_bound_problem']),
            num_gridpoints=grid_dict.get('num_grid_problem', self.default_grid['num_grid_problem']),
        )
        mixing_grid = self.make_grid_list(
            lower_bound=grid_dict.get('lowerBoundMixing', self.default_grid['lowerBoundMixing']),
            upper_bound=grid_dict.get('upperBoundMixing', self.default_grid['upperBoundMixing']),
            num_gridpoints=grid_dict.get('num_grid_mixing', self.default_grid['num_grid_mixing']),
        )
        return problem_grid, mixing_grid

    def make_grid_list(self, lower_bound, upper_bound, num_gridpoints):
        """
        takes lower and upper bound and returns a list of equidistant points in that interval
        with length equal the specified grid points. a non-positive number of grid points will raise an exception.
        If exactly one gridpoint is specified, the returned list will only contain the lower bound as the
        singe point.
    
        Args:
            lower_bound: (float) the minimal grid point
            upper_bound: (float) the maximal grid point
            num_gridpoints: (int) the total number of grid points
        Returns:
            (list) a list of floating values which form a grid on the given intervall
        """
        if num_gridpoints <= 0:
            raise ValueError("trying to construct an empty grid set, which vanishes in the product of grids")
        try:
            step_size = float(upper_bound - lower_bound) / (num_gridpoints - 1)
        except ZeroDivisionError:
            return [lower_bound]
        return [lower_bound + idx * step_size for idx in range(num_gridpoints)]


class QaoaQiskit(BackendBase):
    """
    A class for solving the unit commitment problem using QAOA. This is
    done by constructing the problem internally and then using IBM's
    qiskit package to solve the created problem on simulated or physical
    Hardware.
    """

    def __init__(self, reader: InputReader):
        """
        Constructor for the QaoaQiskit class. It requires an
        InputReader, which handles the loading of the network and
        configuration file.

        Args:
            reader: (InputReader)
                 Instance of an InputReader, which handled the loading
                 of the network and configuration file.
        """
        super().__init__(reader)
        # copy relevant config to make code more readable
        self.config_qaoa = self.config["QaoaBackend"]
        self.add_results_dict()
        self.angle_supervisior = QaoaAngleSupervisor.make_angle_supervisior(
            qaoa_optimizer=self
        )
        self.numAngles = self.angle_supervisior.get_num_angles()

        # initiate local parameters
        self.iteration_counter = None
        self.iter_result = {}
        self.rep_result = {}
        self.qc = None
        self.param_vector = None
        self.statistics = {"confidence": 0.0,
                           "best_bitstring": "",
                           "probabilities": {},
                           "p_values": {},
                           "u_values": {}
                           }

        # set up connection to IBMQ servers
        if self.config_qaoa["noise"] or (not self.config_qaoa["simulate"]):
            IBMQ.save_account(token=self.config["APItoken"]["IBMQ_API_token"],
                              overwrite=True)
            self.provider = IBMQ.load_account()

    def transform_problem_for_optimizer(self) -> None:
        """
        Initializes an ising_interface-instance, which encodes the Ising
        Spin Glass Problem, using the network to be optimized.

        Returns:
            (None)
                Add the ising_interface-instance to
                self.transformed_problem.
        """
        self.transformed_problem = IsingBackbone.build_ising_problem(
            network=self.network, config=self.config["ising_interface"]
        )
        self.output["results"]["qubit_map"] = \
            self.transformed_problem.get_qubit_mapping()

    def optimize(self) -> None:
        """
        Optimizes the network encoded in the ising_interface-instance. A
        self-written Qaoa algorithm is used, which can either simulate
        the quantum part or solve it on one of IBMQ's servers (given the
        correct credentials are provided).
        As classic solvers SPSA, COBYLA or ADAM can be chosen.

        Returns:
            (None)
                The optimized solution is stored in the self.output
                dictionary.
        """
        # retrieve various parameters from the config
        shots = self.config_qaoa["shots"]
        simulator = self.config_qaoa["simulator"]
        simulate = self.config_qaoa["simulate"]
        noise = self.config_qaoa["noise"]
        initial_guess_original = np.array([0 for _ in range(self.angle_supervisior.get_num_angles())])
        num_vars = self.numAngles
        max_iter = self.config_qaoa["max_iter"]
        total_repetition = 0

        hamiltonian = self.transformed_problem.get_hamiltonian_matrix()
        scaled_hamiltonian = self.scale_hamiltonian(hamiltonian=hamiltonian)
        self.output["results"]["hamiltonian"]["original"] = hamiltonian
        self.output["results"]["hamiltonian"]["scaled"] = scaled_hamiltonian
        num_qubits = len(hamiltonian)

        # create ParameterVector to be used as placeholder when creating the quantum circuit
        self.param_vector = ParameterVector("theta", self.numAngles)
        self.qc = self.create_qc(hamiltonian=scaled_hamiltonian, theta=self.param_vector)
        # bind variables beta and gamma to qc, to generate a circuit which is saved in output as latex source code.
        draw_theta = self.create_draw_theta(theta=initial_guess_original)
        qc_draw = self.qc.bind_parameters({self.param_vector: draw_theta})
        self.output["results"]["qc"] = qc_draw.draw(output="latex_source")

        # setup IBMQ backend and save its configuration to output
        backend, noise_model, coupling_map, basis_gates = self.setup_backend(
            simulator=simulator,
            simulate=simulate,
            noise=noise,
            nqubits=num_qubits
        )
        self.output["results"]["backend"] = backend.configuration().to_dict()

        current_repetition = 1
        for initial_guess in self.angle_supervisior.get_initial_angle_iterator():
            time_start = datetime.timestamp(datetime.now())
            total_repetition = current_repetition
            print(
                f"----------------------- Repetition {total_repetition} ----------------------------------"
            )

            self.prepare_repetition_dict()
            self.rep_result["initial_guess"] = initial_guess.tolist()

            self.iteration_counter = 0

            expectation = self.get_expectation(
                backend=backend,
                noise_model=noise_model,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                shots=shots,
                simulate=simulate,
            )

            optimizer = self.get_classical_optimizer(max_iter)

            res = optimizer.optimize(
                num_vars=num_vars,
                objective_function=expectation,
                initial_point=initial_guess,
            )
            self.rep_result["optimized_result"] = {
                "x": list(res[0]),  # solution [beta, gamma]
                "fun": res[1],  # objective function value
                "counts": self.rep_result["iterations"][res[2]]["counts"],  # counts of the optimized result
                "nfev": res[2],  # number of objective function calls
            }

            time_end = datetime.timestamp(datetime.now())
            duration = time_end - time_start
            self.rep_result["duration"] = duration

            self.output["results"]["repetitions"][total_repetition] = self.rep_result

            current_repetition += 1

        self.output["results"]["total_reps"] = total_repetition

    def process_solution(self) -> None:
        """
        Post-processing of the solution. Adds the components from the
        ising_interface-instance to self.output. Furthermore, a
        statistical analysis of the results is performed, to determine,
        if a solution can be found with confidence.

        Returns:
            (None)
                Modifies self.output dictionary with post-process
                information.
        """
        self.output["components"] = self.transformed_problem.get_data()

        self.extract_p_values()  # get probabilities of bitstrings
        self.find_best_bitstring()
        self.compare_bit_string_to_rest()  # one-sided Mann-Witney U Test
        self.determine_confidence()  # check p against various alphas

        self.output["results"]["statistics"] = self.statistics

        self.write_report_to_output(best_bitstring=self.statistics[
            "best_bitstring"])

    def transform_solution_to_network(self) -> None:
        """
        Encodes the optimal solution found during optimization and
        stored in self.output into a pypsa.Network. It reads the
        solution stored in the optimizer instance, prints some
        information regarding it to stdout and then writes it into a
        network, which is then saved in self.output.

        Returns:
            (None)
                Modifies self.output with the output_network.
        """
        self.print_report()
        best_bitstring = self.output["results"]["statistics"]["best_bitstring"]
        solution = []
        for idx, bit in enumerate(best_bitstring):
            if bit == "1":
                solution.append(idx)
        output_network = self.transformed_problem.set_output_network(
            solution=solution)
        output_dataset = output_network.export_to_netcdf()
        self.output["network"] = output_dataset.to_dict()

    def add_results_dict(self) -> None:
        """
        Adds the basic structure for the self.output["results"]
        dictionary.

        Returns:
            (None)
                Modifies self.output["results"].
        """
        self.output["results"] = {
            "backend": None,
            "qubit_map": {},
            "hamiltonian": {},
            "qc": None,
            "initial_guesses": {
                "original": self.config_qaoa["initial_guess"],
                "refined": [],
            },
            "kirchhoff": {},
            "statistics": {},
            "total_reps": 0,
            "repetitions": {},
        }

    def prepare_repetition_dict(self) -> None:
        """
        Initializes the basic structure for the
        self.rep_result-dictionary. Its values are initialized to empty
        dictionaries, empty lists or None values.

        Returns:
            (None)
                Modifies self.rep_result.
        """
        self.rep_result = {
            "initial_guess": [],
            "duration": None,
            "optimized_result": {},
            "iterations": {},
        }

    def prepare_iteration_dict(self) -> None:
        """
        Initializes the basic structure for the
        self.iter_result-dictionary. Its values are initialized to empty
        dictionaries, empty lists or None values.

        Returns:
            (None)
                Modifies self.rep_result.
        """
        self.iter_result = {
            "theta": [],
            "counts": {},
            "bitstrings": {},
            "return": None
        }

    def create_draw_theta(self, theta: list) -> list:
        """
        Creates a list of the same size as theta with Parameters β and γ
        as values. This list can then be used to bind to a quantum
        circuit using Qiskit's bind_parameters function. It can then be
        saved as a generic circuit, using Qiskit's draw function and
        stored for later use or visualization.

        Args:
            theta: (list)
                The list of optimizable values of the quantum circuit.
                It will be used to create a list of the same length with
                β's and γ's.

        Returns:
            (list)
                The created list of β's and γ's.
        """
        beta_values = theta[::2]
        draw_theta = []
        for layer, _ in enumerate(beta_values):
            draw_theta.append(Parameter(f"\u03B2{layer + 1}"))  # append beta_i
            draw_theta.append(Parameter(f"\u03B3{layer + 1}"))  # append gamma_i

        return draw_theta

    def get_classical_optimizer(self,
                                max_iter: int
                                ) -> qiskit.algorithms.optimizers:
        """
        Initiates and returns the classical optimizer set in the config
        file. If another optimizer than SPSA, COBYLA or Adam is
        specified an error is thrown.

        Args:
            max_iter: (int)
                The maximum number of iterations the classical optimizer
                is allowed to perform.

        Returns:
            (qiskit.algorithms.optimizers)
                The initialized classical optimizer, as specified in the
                config.
        """
        config_string = self.config_qaoa["classical_optimizer"]
        if config_string == "SPSA":
            return SPSA(maxiter=max_iter, blocking=False)
        elif config_string == "COBYLA":
            return COBYLA(maxiter=max_iter, tol=0.0001)
        elif config_string == "ADAM":
            return ADAM(maxiter=max_iter, tol=0.0001)
        raise ValueError(
            "Optimizer name in config file doesn't match any known optimizers"
        )

    def get_min_cf_vars(self) -> list:
        """
        Searches through the optimization results of all repetitions
        done with random (or partly random) initial values for beta and
        gamma and finds the minimum cost function value. The associated
        beta and gamma values will be returned as a list to be used in
        the refinement phase of the optimization.

        Returns:
            (list)
                The beta and gamma values associated with the minimal
                cost function value.
        """
        search_data = self.output["results"]["repetitions"]
        min_cf = search_data[1]["optimized_result"]["fun"]
        min_x = []
        for i in range(1, len(search_data) + 1):
            if search_data[i]["optimized_result"]["fun"] <= min_cf:
                min_cf = search_data[i]["optimized_result"]["fun"]
                min_x = search_data[i]["optimized_result"]["x"]

        return min_x

    def scale_hamiltonian(self, hamiltonian: list) -> list:
        """
        Scales the hamiltonian so that the maximum absolute value in the
        input hamiltonian is equal to Pi.

        Args:
            hamiltonian: (list)
                The input hamiltonian to be scaled.

        Returns:
            (list)
                The scaled hamiltonian.
        """
        matrix_max = np.max(hamiltonian)
        matrix_min = np.min(hamiltonian)
        matrix_extreme = max(abs(matrix_max), abs(matrix_min))
        factor = matrix_extreme / math.pi
        scaled_hamiltonian = np.array(hamiltonian) / factor
        return scaled_hamiltonian.tolist()

    def create_qc(self,
                  hamiltonian: list,
                  theta: ParameterVector
                  ) -> QuantumCircuit:
        """
        Creates a qiskit quantum circuit based on the hamiltonian matrix
        given. The quantum circuit will be created using a
        ParameterVector to create placeholders, which can be filled with
        the actual parameters using qiskit's bind_parameters function.

        Args:
            hamiltonian: (dict)
                The matrix representing the problem Hamiltonian.
            theta: (ParameterVector)
                The ParameterVector of the same length as the list of
                optimizable parameters.

        Returns:
            (QuantumCircuit)
                The created quantum circuit.
        """
        num_qubits = len(hamiltonian)
        qc = QuantumCircuit(num_qubits)

        # beta parameters are at even indices and gamma at odd indices
        beta_values = theta[::2]
        gamma_values = theta[1::2]

        # add Hadamard gate to each qubit
        for i in range(num_qubits):
            qc.h(i)

        for layer, _ in enumerate(beta_values):
            # for visual purposes only, when the quantum circuit is drawn
            qc.barrier()
            qc.barrier()
            # add problem Hamiltonian
            for i in range(len(hamiltonian)):
                for j in range(i, len(hamiltonian[i])):
                    if hamiltonian[i][j] != 0.0:
                        if i == j:
                            qc.rz(
                                -hamiltonian[i][j] * gamma_values[layer], i
                            )
                            # inversed, as the implementation in the
                            # ising_interface inverses the values
                        else:
                            qc.rzz(
                                -hamiltonian[i][j] * gamma_values[layer], i, j
                            )
                            # inversed, as the implementation in the
                            # ising_interface inverses the values
            qc.barrier()

            # add mixing Hamiltonian to each qubit
            for i in range(num_qubits):
                qc.rx(beta_values[layer], i)

        qc.measure_all()

        return qc

    def kirchhoff_satisfied(self, bitstring: str) -> float:
        """
        Checks if the kirchhoff constraints are satisfied for the given
        solution.

        Args:
            bitstring: (str)
                The bitstring encoding a possible solution to the
                network.

        Returns:
            (float)
                The absolut deviation from the optimal (0), where the
                kirchhoff constrains would be completely satisfied for
                the given network.
        """
        try:
            # check if the kirchhoff costs for this bitstring have already
            # been calculated and if so use this value
            return self.output["results"]["kirchhoff"][bitstring]["total"]
        except KeyError:
            self.output["results"]["kirchhoff"][bitstring] = {}
            kirchhoff_cost = 0.0
            # calculate the deviation from the optimal for each bus separately
            for bus in self.network.buses.index:
                bitstring_to_solution = [
                    idx for idx, bit in enumerate(bitstring) if bit == "1"
                ]
                for _, val in self.transformed_problem.calc_power_imbalance_at_bus(
                        bus, bitstring_to_solution
                ).items():
                    # store the penalty for each bus and then add them to the
                    # total costs
                    self.output["results"]["kirchhoff"][bitstring][bus] = val
                    kirchhoff_cost += abs(val) ** 2
            self.output["results"]["kirchhoff"][bitstring]["total"] = \
                kirchhoff_cost
            return kirchhoff_cost

    def compute_expectation(self, counts: dict) -> float:
        """
        Computes expectation values based on the measurement/simulation
        results.

        Args:
            counts: (dict)
                The number of times a specific bitstring was measured.
                The bitstring is the key and its count the value.

        Returns:
            (float)
                The expectation value.
        """

        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.kirchhoff_satisfied(bitstring=bitstring)
            avg += obj * count
            sum_count += count
            self.iter_result["bitstrings"][bitstring] = {
                "count": count,
                "obj": obj,
                "avg": avg,
                "sum_count": sum_count,
            }

        self.iter_result["return"] = avg / sum_count
        self.rep_result["iterations"][self.iteration_counter] = self.iter_result

        return self.iter_result["return"]

    def setup_backend(self,
                      simulator: str,
                      simulate: bool,
                      noise: bool,
                      nqubits: int
                      ) -> [BaseBackend, NoiseModel, list, list]:
        """
        Sets up the qiskit backend based on the settings passed into
        the function.

        Args:
            simulator: (str)
                The name of the Quantum Simulator to be used, if
                simulate is True.
            simulate: (bool)
                If True, the specified Quantum Simulator will be used to
                execute the Quantum Circuit. If False, the least busy
                IBMQ Quantum Computer will be initiated to be used to
                execute the Quantum Circuit.
            noise: (bool)
                If True, noise will be added to the Simulator. If False,
                no noise will be added. Only works if "simulate" is set
                to True. On actual IBMQ devices noise is always present
                and cannot be deactivated.
            nqubits: (int)
                The number of Qubits of the Quantum Circuit. Used to
                find a suitable IBMQ Quantum Computer.

        Returns:
            (BaseBackend)
                The backend to be used.
            (NoiseModel)
                The noise model of the chosen backend, if noise is set
                to True. Otherwise it is set to None.
            (list)
                The coupling map of the chosen backend, if noise is set
                to True. Otherwise it is set to None.
            (list)
                The initial basis gates used to compile the noise model,
                if noise is set to True. Otherwise it is set to None.
        """
        noise_model = None
        coupling_map = None
        basis_gates = None
        if simulate:
            if noise:
                # https://qiskit.org/documentation/apidoc/aer_noise.html
                # set IBMQ server to extract noise model and coupling_map
                device = self.provider.get_backend("ibmq_lima")
                # Get noise model from IBMQ server
                noise_model = NoiseModel.from_backend(device)
                # Get coupling map from backend
                coupling_map = device.configuration().coupling_map
                # Get the basis gates for the noise model
                basis_gates = noise_model.basis_gates
                # Select the QasmSimulator from the Aer provider
                backend = Aer.get_backend(simulator)

            else:
                backend = Aer.get_backend(simulator)
        else:
            large_enough_devices = self.provider.backends(
                filters=lambda x: x.configuration().n_qubits > nqubits and not x.configuration().simulator
            )
            backend = least_busy(large_enough_devices)
            # backend = self.provider.get_backend("ibmq_lima")
        return backend, noise_model, coupling_map, basis_gates

    def get_expectation(
            self,
            backend: BaseBackend,
            noise_model: NoiseModel = None,
            coupling_map: list = None,
            basis_gates: list = None,
            shots: int = 1024,
            simulate: bool = True,
    ) -> callable:
        """
        Builds the objective function, which can be used in a classical
        solver.

        Args:
            backend: (BaseBackend)
                The backend to be used.
            noise_model: (NoiseModel)
                The noise model of the chosen backend. Default: None
            coupling_map: (list)
                The coupling map of the chosen backend Default: None
            basis_gates: (list)
                The initial basis gates used to compile the noise model.
                Default: None
            shots: (int)
                The number of repetitions of each circuit, for sampling.
                Default: 1024
            simulate: (bool)
                If True, the specified Quantum Simulator will be used to
                execute the Quantum Circuit. If False, the IBMQ Quantum
                Computer set in setup_backend will be used to execute
                the Quantum Circuit. Default: True

        Returns:
            (callable)
                The objective function to be used in a classical solver.
        """

        def execute_circ(theta):
            qc = self.qc.bind_parameters({self.param_vector: theta})

            if simulate:
                # Run on chosen simulator
                results = execute(
                    experiments=qc,
                    backend=backend,
                    shots=shots,
                    noise_model=noise_model,
                    coupling_map=coupling_map,
                    basis_gates=basis_gates,
                ).result()
            else:
                # Submit job to real device and wait for results
                job_device = execute(experiments=qc,
                                     backend=backend,
                                     shots=shots)
                job_monitor(job_device)
                results = job_device.result()
            counts = results.get_counts()
            self.iteration_counter += 1
            self.prepare_iteration_dict()
            self.iter_result["theta"] = list(theta)
            self.iter_result["counts"] = counts

            return self.compute_expectation(counts=counts)

        return execute_circ

    def extract_p_values(self) -> None:
        """
        Searches through the results and combines the probability for
        each bitstring in each repetition of the "optimized_result" in
        lists. E.g. For 100 repetitions, a list for each bitstring is
        created, containing the 100 probability values (one from each
        repetition)

        Returns:
            (None)
                Writes the created lists of probabilities into the
                self.statistics dictionary.
        """
        data = self.output["results"]["repetitions"]
        bitstrings = self.output["results"]["kirchhoff"].keys()
        probabilities = {}
        shots = self.config_qaoa["shots"]
        # find repetition value from where the refinement process started
        start = self.output["results"]["totalReps"] - self.config_qaoa["repetitions"]
        for bitstring in bitstrings:
            probabilities[bitstring] = []
            for key in data:
                if key <= start:
                    continue
                if bitstring in data[key]["optimized_result"]["counts"]:
                    p = data[key]["optimized_result"][
                            "counts"][bitstring] / shots
                else:
                    p = 0
                probabilities[bitstring].append(p)
        self.statistics["probabilities"] = probabilities

    def find_best_bitstring(self) -> None:
        """
        Takes the median of the probabilities of each bitstring and
        determines, which bitstring has objectively the highest
        probability. This bitstring is stored in self.statistics.

        Returns:
            (None)
                Modifies the self.statistics dictionary.
        """
        probabilities = self.statistics["probabilities"]
        # set first bitstring as best for now
        best_bitstring = list(probabilities.keys())[0]
        # get median of first bitstring
        best_median = median(probabilities[best_bitstring])

        for bitstring in probabilities:
            if median(probabilities[bitstring]) > best_median:
                best_median = median(probabilities[bitstring])
                best_bitstring = bitstring
        self.statistics["best_bitstring"] = best_bitstring

    def compare_bit_string_to_rest(self) -> None:
        """
        Compares the best_bitstring (found in the find_best_bitstring
        function) to every other bitstring using a one-sided Mann
        Whitney U Test, where the alternative hypothesis is that the
        probability to find the best_bitstring is greater than the
        probabilities of the other bitstrings. The results of the tests
        are stored in the self.statistics dictionary.

        Returns:
            (None)
                Modifies the self.statistics dictionary.
        """
        best_bitstring = self.statistics["best_bitstring"]
        probabilities = self.statistics["probabilities"]
        for bitstring in probabilities.keys():
            if bitstring == best_bitstring:
                continue
            u, p = stats.mannwhitneyu(x=probabilities[best_bitstring],
                                      y=probabilities[bitstring],
                                      alternative="greater")
            self.statistics["p_values"][
                f"{best_bitstring}-{bitstring}"] = float(p)
            self.statistics["u_values"][
                f"{best_bitstring}-{bitstring}"] = float(u)

    def determine_confidence(self) -> None:
        """
        Determines with which confidence, if any, the best_bitstring can
        be found. A list of alphas is checked, starting at 0.01 up until
        0.5. The found confidence is then stored in self.statistics. If
        none is found the value in self.statistics["confidence"] is kept
        at 0.0, thus indicating no best_bitstring can be confidently
        determined.

        Returns:
            (None)
                Modifies the self.statistics dictionary.
        """
        alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        for alpha in alphas:
            broken = False
            for key, value in self.statistics["p_values"].items():
                if value > alpha:
                    broken = True
                    break
            if not broken:
                self.statistics["confidence"] = 1 - alpha
                break

    def write_report_to_output(self, best_bitstring: str) -> None:
        """
        Writes solution specific values of the optimizer result and the
        Ising spin glass problem solution to the output dictionary.

        Args:
            best_bitstring: (str)
                The bitstring representing the best solution found
                during optimization.

        Returns:
            (None)
                Modifies self.output with solution specific parameters
                and values.
        """
        solution = []
        for idx, bit in enumerate(best_bitstring):
            if bit == "1":
                solution.append(idx)
        report = self.transformed_problem.generate_report(solution=solution)
        for key in report:
            self.output["results"][key] = report[key]

    def print_solver_specific_report(self):
        """
        Prints a table containing information about all qaoa optimization repetitions that were performed.
        This consists of the repetition number, the score of that repetition and which angles lead to that
        score. The table is sorted by scored and rounded to two decimal places
    
        Returns:
            (None) prints qaoa repetition information
        """
        print("\n--- Table of Results ---")
        repetitions = self.output["results"]["repetitions"]
        repetition_index_sorted_by_score = sorted(
            list(range(1, len(repetitions) + 1)),
            key=lambda x: repetitions[x]['optimized_result']['fun']
        )
        current_score_bracket = 0
        horizontal_break = "------------+---------+--" + self.numAngles * "------"

        # table header
        print(" Repetition |  Score  |" + self.numAngles * "  " + "Solution ")

        for repetition in repetition_index_sorted_by_score:
            repetition_result = self.output["results"]["repetitions"][repetition]
            rounded_angle_solution = [round(angle, 2) for angle in repetition_result['optimized_result']['x']]
            score = repetition_result['optimized_result']['fun']
            # print breaks every integer step
            if score > current_score_bracket:
                print(horizontal_break)
                current_score_bracket = int(score) + 1
            score_str = str(round(score, 2))
            print(f"     {repetition: <7}|  {score_str: <7}|  {rounded_angle_solution}")
