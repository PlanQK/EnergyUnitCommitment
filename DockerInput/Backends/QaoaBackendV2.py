import copy
import json, yaml
import math

import numpy as np
import pypsa
import os.path

from numpy import random

from .IsingPypsaInterface import IsingBackbone                      # import for Docker run
from .BackendBase import BackendBase                                # import for Docker run
#from IsingPypsaInterface import IsingBackbone                      # import for local/debug run
#from BackendBase import BackendBase                                # import for local/debug run
#from EnvironmentVariableManager import EnvironmentVariableManager  # import for local/debug run
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.algorithms.optimizers import SPSA, COBYLA, ADAM
from qiskit.circuit import Parameter


class QaoaQiskit(BackendBase):
    def __init__(self, *args):
        super().__init__(args)
        self.output["results"] = {"backend": None,
                                  "qubit_map": {},
                                  "hamiltonian": {},
                                  "qc": None,
                                  "initial_guesses": {"original": self.config["QaoaBackend"]["initial_guess"],
                                                      "refined": []},
                                  "kirchhoff": {},
                                  "repetitions": {}}
        self.isingInterface = IsingBackbone.buildIsingProblem(network=self.network, config=self.config["IsingInterface"])
        self.iterationCounter = None
        self.totalRepetition = None

        self.kirchhoff = {}
        self.components = {}

        self.docker = os.environ.get("RUNNING_IN_DOCKER", False)
        if self.adapter.config["QaoaBackend"]["noise"] or (not self.adapter.config["QaoaBackend"]["simulate"]):
            IBMQ.save_account(self.adapter.config["APItoken"]["IBMQ_API_token"], overwrite=True)
            self.provider = IBMQ.load_account()

    def prepareRepetitionDict(self):
        self.output["results"]["repetitions"][self.totalRepetition] = {"init_guess": [],
                                                                       "iteration": {},
                                                                       "duration": None,
                                                                       "optimizeResults": {}}

    def prepareIterationDict(self):
        self.output["results"]["repetitions"][self.totalRepetition]["iteration"][self.iterationCounter] = \
            {"theta": [],
             "counts": {},
             "bitstrings": {},
             "return": None}

    def transformProblemForOptimizer(self, network):
        self.output["results"]["qubit_map"] = self.isingInterface.getQubitMapping()

        return self.isingInterface

    def transformSolutionToNetwork(self, network, transformedProblem, solution):
        return network

    def processSolution(self, network, transformedProblem, solution):
        return solution

    def createDrawTheta(self, theta: list) -> list:
        betaValues = theta[::2]
        drawTheta = []
        for layer, _ in enumerate(betaValues):
            drawTheta.append(f"{chr(946)}{chr(8320 + layer)}")  #append beta_i
            drawTheta.append(f"{chr(947)}{chr(8320 + layer)}")  #append gamma_i

        return drawTheta

    def optimize(self, transformedProblem):

        shots = self.config["QaoaBackend"]["shots"]
        simulator = self.config["QaoaBackend"]["simulator"]
        simulate = self.config["QaoaBackend"]["simulate"]
        noise = self.config["QaoaBackend"]["noise"]
        initial_guess_original = copy.deepcopy(self.config["QaoaBackend"]["initial_guess"])
        num_vars = len(initial_guess_original)
        max_iter = self.config["QaoaBackend"]["max_iter"]
        repetitions = self.config["QaoaBackend"]["repetitions"]

        if "rand" in initial_guess_original:
            randRep = 2
        else:
            randRep = 1

        hamiltonian = transformedProblem.getHamiltonianMatrix()
        scaledHamiltonian = self.scaleHamiltonian(hamiltonian=hamiltonian)
        self.output["results"]["hamiltonian"]["original"] = hamiltonian
        self.output["results"]["hamiltonian"]["scaled"] = scaledHamiltonian

        drawTheta = self.createDrawTheta(theta=initial_guess_original)
        qcDraw = self.create_qc(hamiltonian=scaledHamiltonian, theta=drawTheta)
        self.output["results"]["qc"] = qcDraw.draw(output="latex_source")

        for rand in range(randRep):
            for curRepetition in range(1, repetitions + 1):
                time_start = datetime.timestamp(datetime.now())
                self.totalRepetition = rand * repetitions + curRepetition
                print(f"----------------------- Repetition {self.totalRepetition} ----------------------------------")

                initial_guess = []
                for j in range(num_vars):
                    # choose initial guess randomly (between 0 and 2PI for beta and 0 and PI for gamma)
                    if initial_guess_original[j] == "rand":
                        if j % 2 == 0:
                            initial_guess.append((0.5 - random.rand()) * 2 * math.pi)
                        else:
                            initial_guess.append((0.5 - random.rand()) * math.pi)
                    else:
                        initial_guess.append(initial_guess_original[j])
                initial_guess = np.array(initial_guess)

                self.prepareRepetitionDict()
                self.output["results"]["repetitions"][self.totalRepetition]["initial_guess"] = initial_guess.tolist()

                filename = self.generateFilename(self.totalRepetition)
                self.iterationCounter = 0

                expectation = self.get_expectation(nqubits=,
                                                   simulator=simulator,
                                                   shots=shots,
                                                   simulate=simulate,
                                                   noise=noise)

                optimizer = self.getClassicalOptimizer(max_iter)

                res = optimizer.optimize(num_vars=num_vars, objective_function=expectation, initial_point=initial_guess)
                self.output["results"]["repetitions"][self.totalRepetition] \
                    ["optimizeResults"] = {"x": list(res[0]),  # solution [beta, gamma]
                                           "fun": res[1],  # objective function value
                                           "nfev": res[2]}  # number of objective function calls

                time_end = datetime.timestamp(datetime.now())
                duration = time_end - time_start
                self.output["results"]["repetitions"][self.totalRepetition]["duration"] = duration

            if "rand" in initial_guess_original:
                minCFvars = self.getMinCFvars()
                self.output["results"]["initial_guesses"]["refined"] = minCFvars
                for j in range(num_vars):
                    if initial_guess_original[j] == "rand":
                        initial_guess_original[j] = minCFvars[j]

        return self.output["results"]


    def generateFilename(self, currentRepetitionNumber):

        filename_input = str(self.config["QaoaBackend"]["filenameSplit"][1]) + "_" + \
                         str(self.config["QaoaBackend"]["filenameSplit"][2]) + "_" + \
                         str(self.config["QaoaBackend"]["filenameSplit"][3]) + "_" + \
                         str(self.config["QaoaBackend"]["filenameSplit"][4])
        filename_date = str(self.config["QaoaBackend"]["filenameSplit"][7])
        filename_time = str(self.config["QaoaBackend"]["filenameSplit"][8])
        filename_config = str(self.config["QaoaBackend"]["filenameSplit"][9])
        if len(self.config["QaoaBackend"]["filenameSplit"]) == 11:
            filename_config += "_" + str(self.config["QaoaBackend"]["filenameSplit"][10])

        filename = "_".join([
                "Qaoa", filename_input, filename_date, filename_time, \
                filename_config, "", str(currentRepetitionNumber)
        ]) 
        return filename + ".json"
    

    def getClassicalOptimizer(self, max_iter):
        configString = self.config["QaoaBackend"]["classical_optimizer"]
        if configString == "SPSA":
            return SPSA(maxiter=max_iter, blocking=False)
        elif configString == "COBYLA":
            return COBYLA(maxiter=max_iter, tol=0.0001)
        elif configString == "ADAM":
            return ADAM(maxiter=max_iter, tol=0.0001)
        raise ValueError("Optimizer name in config file doesn't match any known optimizers")


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

    def getMetaInfo(self):
        return self.metaInfo

    def validateInput(self, path, network):
        pass

    def handleOptimizationStop(self, path, network):
        pass

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

    def create_qc(self, hamiltonian: dict, theta: list) -> QuantumCircuit:
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
        qc.barrier()

        for layer, _ in enumerate(betaValues):
            # add problem Hamiltonian
            for i in range(len(hamiltonian)):
                for j in range(i, len(hamiltonian[i])):
                    if hamiltonian[i][j] != 0.0:
                        if i == j:
                            qc.rz(-hamiltonian[i][j] * gammaValues[layer], i)  # negative because it´s the inverse of original QC
                        else:
                            qc.rzz(-hamiltonian[i][j] * gammaValues[layer], i, j)  # negative because it´s the inverse of original QC
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
                bitstringToSolution = [idx for idx, bit in enumerate(bitstring) if bit == "1" ]
                for _, val in self.isingInterface.calcPowerImbalanceAtBus(bus, bitstringToSolution).items():
                    self.output["results"]["kirchhoff"][bitstring][bus] = val
                    kirchhoffCost += abs(val) ** 2
            # TODO seperate into different cost functions
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
            self.output["results"]["repetitions"][self.totalRepetition]["iterations"][self.iterationCounter] \
                ["bitstrings"][bitstring] = {"count": count,
                                             "obj": obj,
                                             "avg": avg,
                                             "sum_count": sum_count}

        self.output["results"]["repetitions"][self.totalRepetition]["iterations"][self.iterationCounter] \
            ["return"] = avg / sum_count

        return avg / sum_count

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
                    filters=lambda x: x.configuration().n_qubits > nqubits and not x.configuration().simulator)
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
                filters=lambda x: x.configuration().n_qubits > nqubits and not x.configuration().simulator)
            backend = least_busy(large_enough_devices)
            # backend = provider.get_backend("ibmq_lima")
            noise_model = None
            coupling_map = None
            basis_gates = None

        return backend, noise_model, coupling_map, basis_gates

    def get_expectation(self, nqubits: int, simulator: str = "aer_simulator",
                        shots: int = 1024, simulate: bool = True, noise: bool = False):
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
        backend, noise_model, coupling_map, basis_gates = self.setup_backend(simulator=simulator,
                                                                             simulate=simulate,
                                                                             noise=noise,
                                                                             nqubits=nqubits)

        self.output["results"]["backend"] = backend.configuration().to_dict()

        def execute_circ(theta):
            qc = self.create_qc(hamiltonian=self.output["results"]["hamiltonian"]["scaled"], theta=theta)

            if simulate:
                # Run on chosen simulator
                results = execute(experiments=qc,
                                  backend=backend,
                                  shots=shots,
                                  noise_model=noise_model,
                                  coupling_map=coupling_map,
                                  basis_gates=basis_gates).result()
            else:
                # Submit job to real device and wait for results
                job_device = execute(experiments=qc,
                                     backend=backend,
                                     shots=shots)
                job_monitor(job_device)
                results = job_device.result()
            counts = results.get_counts()
            self.iterationCounter += 1
            self.prepareIterationDict()
            self.results_dict[f"rep{self.results_dict['iter_count']}"] = {}
            self.output["results"]["repetitions"][self.totalRepetition]["iterations"][self.iterationCounter]["theta"] = list(theta)
            self.output["results"]["repetitions"][self.totalRepetition]["iterations"][self.iterationCounter]["counts"] = counts

            return self.compute_expectation(counts=counts)

        return execute_circ


def main():
    inputNet = "testNetwork4QubitIsing_2_0_20.nc"
    configFile = "config.yaml"
    outPREFIX = "infoNoCost"
    now = datetime.today()
    outDateTime = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
    outInfo = f"{outPREFIX}_{inputNet}_1_1_{outDateTime}_{configFile}"
    DEFAULT_ENV_VARIABLES = {
        "inputNetwork": inputNet,
        "inputInfo": "",
        "outputNetwork": "",
        "outputInfo": outInfo,
        "outputInfoTime": outDateTime,
        "optimizationCycles": 1000,
        "temperatureSchedule": "[0.1,iF,0.0001]",
        "transverseFieldSchedule": "[10,.1]",
        "monetaryCostFactor": 0.1,
        "kirchhoffFactor": 1.0,
        "slackVarFactor": 70.0,
        "minUpDownFactor": 0.0,
        "trotterSlices": 32,
        "problemFormulation": "binarysplitNoMarginalCost",
        "dwaveAPIToken": "",
        "dwaveBackend": "hybrid_discrete_quadratic_model_version1",
        "annealing_time": 500,
        "programming_thermalization": 0,
        "readout_thermalization": 0,
        "num_reads": 1,
        "chain_strength": 250,
        "strategy": "LowestEnergy",
        "lineRepresentation": 0,
        "postprocess": "flow",
        "timeout": "-1",
        "maxOrder": 0,
        "sampleCutSize": 200,
        "threshold": 0.5,
        "seed": 2
    }

    envMgr = EnvironmentVariableManager(DEFAULT_ENV_VARIABLES)
    netImport = pypsa.Network(os.path.dirname(__file__) + "../../../sweepNetworks/" + inputNet)

    with open(os.path.dirname(__file__) + "/../Configs/" + configFile) as file:
        config = yaml.safe_load(file)

    filenameSplit = str(envMgr['outputInfo']).split("_")
    config["QaoaBackend"]["filenameSplit"] = filenameSplit
    config["QaoaBackend"]["outputInfoTime"] = envMgr["outputInfoTime"]

    qaoa = QaoaQiskitIsing(config=config)
    components = qaoa.transformProblemForOptimizer(network=netImport)

    """
    # https://qiskit.org/documentation/stubs/qiskit.algorithms.QAOA.html
    # https://blog.xa0.de/post/Solving-QUBOs-with-qiskit-QAOA-example/
    # https://qiskit.org/documentation/optimization/stubs/qiskit_optimization.QuadraticProgram.html
    qp = QuadraticProgram()
    #[qp.binary_var() for _ in range(components["hamiltonian"]["scaled"].shape[0])]
    [qp.binary_var() for _ in range(4)]
    qp.minimize(quadratic=components["hamiltonian"]["scaled"])

    quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator'))
    cobyla = COBYLA(maxiter=100)
    qaoaQiskit = QAOA(optimizer=cobyla,reps=10,quantum_instance=quantum_instance)
    #qaoaQiskit = QAOA(quantum_instance=quantum_instance)
    qiskitOpt = MinimumEigenOptimizer(qaoaQiskit)
    qaoa_result = qiskitOpt.solve(qp)

    cobyla = COBYLA(maxiter=100)
    #qaoaQiskit = QAOA(optimizer=cobyla,reps=10,initial_state=qaoa.config["QaoaBackend"]["initial_guess"],quantum_instance=quantum_instance)
    #qaoa_result = qaoaQiskit.find_minimum()
    #qaoa_result = qaoaQiskit.find_minimum(cost_fn=qaoa.get_expectation_QaoaQiskit(counts=20000, components=components, filename="testQaoaQiskit"))
    """
    """
    theta = [Parameter("\u03B2"), Parameter("\u03B3")]
    config["QaoaBackend"]["qcGeneration"] = "IterationMatrix"
    componentsIterM = qaoa.transformProblemForOptimizer(network=netImport)
    qcIterM = qaoa.create_qc1(components=componentsIterM, theta=theta)
    qcIterDrawnM = qcIterM.draw(output="latex_source")
    config["QaoaBackend"]["qcGeneration"] = "Iteration"
    componentsIter = qaoa.transformProblemForOptimizer(network=netImport)
    qcIter = qaoa.create_qc1(components=componentsIter, theta=theta)
    qcIterDrawn = qcIter.draw(output="latex_source")
    config["QaoaBackend"]["qcGeneration"] = "Ising"
    componentsIsing = qaoa.transformProblemForOptimizer(network=netImport)
    qcIsing = qaoa.create_qcIsing(hamiltonian=componentsIsing["hamiltonian"]["scaled"], theta=theta)
    qcIsingDrawn = qcIsing.draw(output="latex_source")

    qcCompare = {"Iteration": qcIterDrawn,
                 "IterationMatrix": qcIterDrawnM,
                 "Ising": qcIsingDrawn}

    with open("qcCompare.json", "w") as write_file:
        json.dump(qcCompare, write_file, indent=2, default=str)
    """

    qaoa.optimize(transformedProblem=components)

    filename = str(envMgr['outputInfo'])
    with open(os.path.dirname(__file__) + "/../../sweepNetworks/" + filename, "w") as write_file:
        json.dump(qaoa.metaInfo["results"], write_file, indent=2, default=str)


if __name__ == "__main__":
    main()
