import copy
import math
import numpy as np
import pypsa

try:
    from .IsingPypsaInterface import IsingBackbone  # import for Docker run
    from .BackendBase import BackendBase  # import for Docker run
except ImportError:
    from IsingPypsaInterface import IsingBackbone  # import for local/debug run
    from BackendBase import BackendBase  # import for local/debug run
from .InputReader import InputReader
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.algorithms.optimizers import SPSA, COBYLA, ADAM
from qiskit.circuit import Parameter, ParameterVector


class QaoaQiskit(BackendBase):
    def __init__(self, reader: InputReader):
        super().__init__(reader)

        # copy relevant config to make code more readable
        self.config_qaoa = self.config["QaoaBackend"]
        self.addResultsDict()

        # initiate local parameters
        self.isingInterface = None
        self.iterationCounter = None
        self.iter_result = {}
        self.rep_result = {}
        self.qc = None
        self.paramVector = None

        # set up connection to IBMQ servers
        if self.config_qaoa["noise"] or (not self.config_qaoa["simulate"]):
            IBMQ.save_account(self.config["APItoken"]["IBMQ_API_token"], overwrite=True)
            self.provider = IBMQ.load_account()

    def addResultsDict(self):
        """
        Adds the basic structure for the self.output["results"]-dictionary.

        Returns:
            Nothing. Modifies the self.output["results"]-dictionary.
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
            "repetitions": {},
        }

    def prepareRepetitionDict(self):
        """
        Initializes the basic structure for the self.rep_result-dictionary, setting its values to empty dictionaries,
        empty lists or None values.

        Returns:
            Nothing. Modifies the self.rep_result-dictionary.
        """
        self.rep_result = {
            "initial_guess": [],
            "iterations": {},
            "duration": None,
            "optimizeResults": {},
        }

    def prepareIterationDict(self):
        """
        Initializes the basic structure for the self.iter_result-dictionary, setting its values to empty dictionaries,
        empty lists or None values.

        Returns:
            Nothing. Modifies the self.iter_result-dictionary.
        """
        self.iter_result = {"theta": [], "counts": {}, "bitstrings": {}, "return": None}

    def transformProblemForOptimizer(self, network: pypsa.Network) -> IsingBackbone:
        """
        Initializes an IsingInterface-instance, which encodes the Ising Spin Glass Problem, using the network to be
        optimized.

        Args:
            network: (pypsa.Network) The network to be optimized.

        Returns:
            (IsingBackbone) The IsingInterface-instance, which encodes the Ising Spin Glass Problem.
        """
        self.isingInterface = IsingBackbone.buildIsingProblem(
            network=self.network, config=self.config["IsingInterface"]
        )
        self.output["results"]["qubit_map"] = self.isingInterface.getQubitMapping()

        return self.isingInterface

    def transformSolutionToNetwork(
            self, network: pypsa.Network, transformedProblem: IsingBackbone, solution: dict
    ) -> pypsa.Network:
        """
        Encodes the optimal solution found during optimization into a pypsa.Network.

        Args:
            network: (pypsa.Network) The network to be optimized.
            transformedProblem: (IsingBackbone) The IsingInterface-instance, which encodes the Ising Spin Glass Problem.
            solution: (dict) The optimal solution to the problem.

        Returns:
            (pypsa.Network) The optimized network.
        """
        return network

    def processSolution(self, network, transformedProblem, solution):
        """
        Post processing of the solution. Adds the components from the IsingInterface-instance to the output.

        Args:
            network: (pypsa.Network) The network to be optimized.
            transformedProblem: (IsingBackbone) The IsingInterface-instance, which encodes the Ising Spin Glass Problem.
            solution: (dict) The optimal solution to the problem.

        Returns:
            (dict) The post-processed solution.
        """
        self.output["components"] = self.isingInterface.getData()
        return self.output

    def createDrawTheta(self, theta: list) -> list:
        """
        Creates a list of the same size as theta with Parameters (beta(s) and gamma(s)) as values. This list can then
        be used to bind to a quantum circuit, using Qiskit's bind_parameters function, which can be saved as a generic
        circuit, using Qiskit's draw function.

        Args:
            theta: (list) The list of optimizable values of the quantum circuit. It will be used to create a list of
                          the same length with beta(s) and gamma(s).

        Returns:
            (list) The created list of beta(s) and gamma(s).
        """
        betaValues = theta[::2]
        drawTheta = []
        for layer, _ in enumerate(betaValues):
            drawTheta.append(Parameter(f"\u03B2{layer+1}"))  # append beta_i
            drawTheta.append(Parameter(f"\u03B3{layer+1}"))  # append gamma_i
            # drawTheta.append(f"{chr(946)}{chr(8320 + layer)}")  #append beta_i
            # drawTheta.append(f"{chr(947)}{chr(8320 + layer)}")  #append gamma_i

        return drawTheta

    def optimize(self, transformedProblem):
        """
        Optimizes the network encoded in the IsingInterface-instance. A self-written Qaoa algorithm is used, which can
        either simulate the quantum part or solve it on one of IBMQ's servers (provided the correct credentials).
        As classic solvers SPSA, COBYLA or ADAM can be chosen.

        Args:
            transformedProblem: (IsingBackbone) The IsingInterface-instance, which encodes the Ising Spin Glass Problem.

        Returns:
            (dict) The optimized solution.
        """
        # retrieve various parameters from the config
        shots = self.config_qaoa["shots"]
        simulator = self.config_qaoa["simulator"]
        simulate = self.config_qaoa["simulate"]
        noise = self.config_qaoa["noise"]
        initial_guess_original = copy.deepcopy(self.config_qaoa["initial_guess"])
        num_vars = len(initial_guess_original)
        max_iter = self.config_qaoa["max_iter"]
        repetitions = self.config_qaoa["repetitions"]

        if "rand" in initial_guess_original:
            randRep = 2
        else:
            randRep = 1

        hamiltonian = transformedProblem.getHamiltonianMatrix()
        scaledHamiltonian = self.scaleHamiltonian(hamiltonian=hamiltonian)
        self.output["results"]["hamiltonian"]["original"] = hamiltonian
        self.output["results"]["hamiltonian"]["scaled"] = scaledHamiltonian
        nqubits = len(hamiltonian)

        # create ParameterVector to be used as placeholder when creating the quantum circuit
        self.paramVector = ParameterVector("theta", len(initial_guess_original))
        self.qc = self.create_qc(hamiltonian=scaledHamiltonian, theta=self.paramVector)
        # bind variables beta and gamma to qc, to generate a circuit which is saved in output as latex source code.
        drawTheta = self.createDrawTheta(theta=initial_guess_original)
        qcDraw = self.qc.bind_parameters({self.paramVector: drawTheta})
        self.output["results"]["qc"] = qcDraw.draw(output="latex_source")

        # setup IBMQ backend and save its configuration to output
        backend, noise_model, coupling_map, basis_gates = self.setup_backend(
            simulator=simulator, simulate=simulate, noise=noise, nqubits=nqubits
        )
        self.output["results"]["backend"] = backend.configuration().to_dict()

        for rand in range(randRep):
            for curRepetition in range(1, repetitions + 1):
                time_start = datetime.timestamp(datetime.now())
                totalRepetition = rand * repetitions + curRepetition
                print(
                    f"----------------------- Repetition {totalRepetition} ----------------------------------"
                )

                initial_guess = []
                for j in range(num_vars):
                    # choose initial guess randomly (between 0 and 2PI for beta and 0 and PI for gamma)
                    if initial_guess_original[j] == "rand":
                        if j % 2 == 0:
                            initial_guess.append((0.5 - np.random.rand()) * 2 * math.pi)
                        else:
                            initial_guess.append((0.5 - np.random.rand()) * math.pi)
                    else:
                        initial_guess.append(initial_guess_original[j])
                initial_guess = np.array(initial_guess)

                self.prepareRepetitionDict()
                self.rep_result["initial_guess"] = initial_guess.tolist()

                self.iterationCounter = 0

                expectation = self.get_expectation(
                    backend=backend,
                    noise_model=noise_model,
                    coupling_map=coupling_map,
                    basis_gates=basis_gates,
                    shots=shots,
                    simulate=simulate,
                )

                optimizer = self.getClassicalOptimizer(max_iter)

                res = optimizer.optimize(
                    num_vars=num_vars,
                    objective_function=expectation,
                    initial_point=initial_guess,
                )
                self.rep_result["optimizeResults"] = {
                    "x": list(res[0]),  # solution [beta, gamma]
                    "fun": res[1],  # objective function value
                    "nfev": res[2],
                }  # number of objective function calls

                time_end = datetime.timestamp(datetime.now())
                duration = time_end - time_start
                self.rep_result["duration"] = duration

                self.output["results"]["repetitions"][totalRepetition] = self.rep_result

            if "rand" in initial_guess_original:
                minCFvars = self.getMinCFvars()
                self.output["results"]["initial_guesses"]["refined"] = minCFvars
                for j in range(num_vars):
                    if initial_guess_original[j] == "rand":
                        initial_guess_original[j] = minCFvars[j]

        return self.output

    def getClassicalOptimizer(self, max_iter):
        configString = self.config_qaoa["classical_optimizer"]
        if configString == "SPSA":
            return SPSA(maxiter=max_iter, blocking=False)
        elif configString == "COBYLA":
            return COBYLA(maxiter=max_iter, tol=0.0001)
        elif configString == "ADAM":
            return ADAM(maxiter=max_iter, tol=0.0001)
        raise ValueError(
            "Optimizer name in config file doesn't match any known optimizers"
        )

    def getMinCFvars(self):
        """
        Searches through metaInfo["results"] and finds the minimum cost function value along with the associated betas
        and gammas, which will be returned as a list.
        Returns:
            minX (list) a list with the betas and gammas associated with the minimal cost function value.
        """
        searchData = self.output["results"]["repetitions"]
        minCF = searchData[1]["optimizeResults"]["fun"]
        minX = []
        for i in range(1, len(searchData) + 1):
            if searchData[i]["optimizeResults"]["fun"] <= minCF:
                minCF = searchData[i]["optimizeResults"]["fun"]
                minX = searchData[i]["optimizeResults"]["x"]

        return minX

    def scaleHamiltonian(self, hamiltonian: list) -> list:
        """
        Scales the hamiltonian so that the maximum absolute value in the input hamiltonian is equal to Pi

        Args:
            hamiltonian: (list) The input hamiltonian to be scaled.

        Returns:
            (list) the scaled hamiltonian.
        """
        matrixMax = np.max(hamiltonian)
        matrixMin = np.min(hamiltonian)
        matrixExtreme = max(abs(matrixMax), abs(matrixMin))
        factor = matrixExtreme / math.pi
        scaledHamiltonian = np.array(hamiltonian) / factor

        return scaledHamiltonian.tolist()

    def create_qc(self, hamiltonian: list, theta: ParameterVector) -> QuantumCircuit:
        """
        Creates a quantum circuit based on the hamiltonian matrix given.

        Args:
            hamiltonian: (dict) The matrix representing the problem Hamiltonian.
            theta: (list) The optimizable values of the quantum circuit. Two arguments needed: beta = theta[0] and
                          gamma = theta[1].

        Returns:
            (QuantumCircuit) The created quantum circuit.
        """
        nqubits = len(hamiltonian)
        qc = QuantumCircuit(nqubits)

        # beta parameters are at even indices and gamma at odd indices
        betaValues = theta[::2]
        gammaValues = theta[1::2]

        # add Hadamard gate to each qubit
        for i in range(nqubits):
            qc.h(i)

        for layer, _ in enumerate(betaValues):
            qc.barrier()
            qc.barrier()
            # add problem Hamiltonian
            for i in range(len(hamiltonian)):
                for j in range(i, len(hamiltonian[i])):
                    if hamiltonian[i][j] != 0.0:
                        if i == j:
                            qc.rz(
                                -hamiltonian[i][j] * gammaValues[layer], i
                            )  # negative because it´s the inverse of original QC
                        else:
                            qc.rzz(
                                -hamiltonian[i][j] * gammaValues[layer], i, j
                            )  # negative because it´s the inverse of original QC
            qc.barrier()

            # add mixing Hamiltonian to each qubit
            for i in range(nqubits):
                qc.rx(betaValues[layer], i)

        qc.measure_all()

        return qc

    def kirchhoff_satisfied(self, bitstring: str) -> float:
        """
        Checks if the kirchhoff constraints are satisfied for the given solution.

        Args:
            bitstring: (str) The possible solution to the network.
            components: (dict) All components to be modeled as a Quantum Circuit.

        Returns:
            (float) The absolut deviation from the optimal (0), where the kirchhoff constrains would be completely
                    satisfied for the given network.
        """
        try:
            return self.output["results"]["kirchhoff"][bitstring]["total"]
        except KeyError:
            self.output["results"]["kirchhoff"][bitstring] = {}
            kirchhoffCost = 0.0
            for bus in self.network.buses.index:
                bitstringToSolution = [
                    idx for idx, bit in enumerate(bitstring) if bit == "1"
                ]
                for _, val in self.isingInterface.calcPowerImbalanceAtBus(
                    bus, bitstringToSolution
                ).items():
                    self.output["results"]["kirchhoff"][bitstring][bus] = val
                    kirchhoffCost += abs(val) ** 2
            self.output["results"]["kirchhoff"][bitstring]["total"] = kirchhoffCost
            return kirchhoffCost

    def compute_expectation(self, counts: dict) -> float:
        """
        Computes expectation value based on measurement results

        Args:
            counts: (dict) The bitstring is the key and its count the value.
            components: (dict) All components to be modeled as a Quantum Circuit.
            filename: (str) The name of the file to which the results shall be safed.

        Returns:
            (float) The expectation value.
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
        self.rep_result["iterations"][self.iterationCounter] = self.iter_result

        return self.iter_result["return"]

    def setup_backend(self, simulator: str, simulate: bool, noise: bool, nqubits: int):
        """
        Sets up the backend based on the settings passed to the function.

        Args:
            simulator: (str) The name of the Quantum Simulator to be used, if simulate is True.
            simulate: (bool) If True, the specified Quantum Simulator will be used to execute the Quantum Circuit.
                             If False, the least busy IBMQ Quantum Comupter will be used to execute the Quantum
                             Circuit.
            noise: (bool) If True, noise will be added to the Simulator. If False, no noise will be added. Only works
                          if "simulate" is set to True.
            nqubits: (int) Number of Qubits of the Quantum Circuit. Used to find a suitable IBMQ Quantum Computer.

        Returns:
            (BaseBackend) The backend to be used.
            (NoiseModel) The noise model of the chosen backend, if noise is set to True.
            (list) The coupling map of the chosen backend, if noise is set to True.
            (NoiseModel.basis_gates) The basis gates of the noise model, if noise is set to True.
        """
        if simulate:
            if noise:
                # https://qiskit.org/documentation/apidoc/aer_noise.html
                large_enough_devices = self.provider.backends(
                    filters=lambda x: x.configuration().n_qubits > nqubits
                    and not x.configuration().simulator
                )
                device = least_busy(large_enough_devices)

                # Get noise model from backend
                noise_model = NoiseModel.from_backend(device)
                # Get coupling map from backend
                coupling_map = device.configuration().coupling_map
                # Get the basis gates for the noise model
                basis_gates = noise_model.basis_gates

                # Select the QasmSimulator from the Aer provider
                backend = Aer.get_backend(simulator)

            else:
                backend = Aer.get_backend(simulator)
                noise_model = None
                coupling_map = None
                basis_gates = None

        else:
            large_enough_devices = self.provider.backends(
                filters=lambda x: x.configuration().n_qubits > nqubits
                and not x.configuration().simulator
            )
            backend = least_busy(large_enough_devices)
            # backend = provider.get_backend("ibmq_lima")
            noise_model = None
            coupling_map = None
            basis_gates = None

        return backend, noise_model, coupling_map, basis_gates

    def get_expectation(
        self,
        backend,
        noise_model,
        coupling_map,
        basis_gates,
        shots: int = 1024,
        simulate: bool = True,
    ):
        """
        Builds the objective function, which can be used in a classical solver.

        Args:
            filename: (str) The name of the file to which the results shall be safed.
            components: (dict) All components to be modeled as a Quantum Circuit.
            simulator: (str) The name of the Quantum Simulator to be used, if simulate is True. Default: "aer_simulator"
            shots: (int) Number of repetitions of each circuit, for sampling. Default: 1024
            simulate: (bool) If True, the specified Quantum Simulator will be used to execute the Quantum Circuit. If
                               False, the least busy IBMQ Quantum Comupter will be used to execute the Quantum Circuit.
                               Default: True
            noise: (bool) If True, noise will be added to the Simulator. If False, no noise will be added. Only works
                            if "simulate" is set to True. Default: False

        Returns:
            (callable) The objective function to be used in a classical solver
        """

        def execute_circ(theta):
            qc = self.qc.bind_parameters({self.paramVector: theta})

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
                job_device = execute(experiments=qc, backend=backend, shots=shots)
                job_monitor(job_device)
                results = job_device.result()
            counts = results.get_counts()
            self.iterationCounter += 1
            self.prepareIterationDict()
            self.iter_result["theta"] = list(theta)
            self.iter_result["counts"] = counts

            return self.compute_expectation(counts=counts)

        return execute_circ
