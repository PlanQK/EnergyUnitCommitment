import copy
import math
from typing import Tuple

import numpy as np
import pypsa
import qiskit
from numpy import median
from scipy import stats

from itertools import product

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
from qiskit.providers import BaseBackend
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.algorithms.optimizers import SPSA, COBYLA, ADAM
from qiskit.circuit import Parameter, ParameterVector



class QaoaParameterSupervisor:
    """a class for choosing qaoa parameters when making (multiple) runs in order to get
    well distributed samples
    """

    @classmethod
    def makeParameterSupervisior(self, **kwargs):
        """
        A factory method for returning the correct supervision for the chosen strategy.
        The "RandomOrFixed" supervision will choose, based on a config list either a fixed
        float value or a random parameter.
        The "GridSearch" will use a grid to evenly distribute intial parameters across the
        parameter space.
    
        Args:
            *args: (list) 
                all the paramters needed to choose and initilize the correct supervisor
        Returns:
            (QaoaParameterSupervisor) An instance of subclass of a QaoaParameterSupervisor
        """
        supervisior_type = kwargs.pop("supervisior_type")
        if supervisior_type == "RandomOrFixed":
            return QaoaParameterSupervisorRandomOrFixed(**kwargs)
        if supervisior_type == "GridSearch":
            return QaoaParameterSupervisorGridSearch(**kwargs)


    def chooseInitalParameters(self):
        """
        For a given qaoa run, returns a list of angle parameters that will be used to start
        a run.

        This method does not accept any parameters. The idea is that calling this method and
        optimizing a circuit will mutate the internal state of this object. The returned value
        will only depend on the state of this object, whose job is to track wich parameters have
        already been chosen to get a good coverage of the parameter space
    
        Returns:
            (list) 
                a list of angle parameters with problem angles on even indices and mixing angles
                at odd indices. The list follows the circuit direction (left to right, counting up)
        """
        raise NotImplementedError

    def getInitialAngleIterator(self):
        raise NotImplementedError
    
    def getNumAngles(self):
        raise NotImplementedError
        
    def getNumRepetitions(self):
        raise NotImplementedError

class QaoaParameterSupervisorRandomOrFixed(QaoaParameterSupervisor):
    """a class for choosing qaoa paramter. The strategy is given by a list. Either an angle parameter
    is fixed based on the list entry, or chosen randomly
    """

    def __init__(self, qaoaOptimizer, param_problem_range=2, param_mixing_range=1):
        """
        Since this list doesn't try some form of grid search, no internal state to track already
        returned parameters is needed.
    
        Args:
            parameter_problem_range: (float) 
                the maximal returned random parameter value for the angle of a problem hamiltonian
                sub-circuit scaled by pi. If this value is 1, the range will be (-pi, pi)
            parameter_mixing_range: (float) 
                the maximal returned random parameter value for the angle of the mixing hamiltonian
                sub-circuit scaled by pi. If this value is 1, the range will be (-pi, pi)
        Returns:
            (list) a list of float values to be used as angles in a qaoa circuit
        """
        self.param_problem_range = param_problem_range
        self.param_mixing_range = param_mixing_range
        self.qaoaOptimizer = qaoaOptimizer
        self.config_guess = qaoaOptimizer.config_qaoa["initial_guess"]
        self.num_vars = len(self.config_guess)
        self.repetitions = self.qaoaOptimizer.config_qaoa["repetitions"]


    def getNumRepetitions(self):
        return self.repetitions 
    

    def getNumAngles(self):
        return self.num_vars


    def getBestInitalParameters(self):
        minCFvars = self.qaoaOptimizer.getMinCFvars()
        self.qaoaOptimizer.output["results"]["initial_guesses"]["refined"] = minCFvars
        bestInitialGuess = self.config_guess
        for j in range(self.num_vars):
            if bestInitialGuess[j] == "rand":
                bestInitialGuess[j] = minCFvars[j]
        return bestInitialGuess


    def getInitialAngleIterator(self):
        for idx in range(self.repetitions):
            yield self.chooseInitalParameters()

        if "rand" not in self.config_guess:
            return

        self.config_guess = self.getBestInitalParameters()
        for idx in range(self.repetitions):
            yield self.chooseInitalParameters()


    def chooseInitalParameters(self):
        """
        Method for return the list of angles to be used
    
        Returns:
            (np.array) an np.array of floats
        """
        inital_angles = []
        for idx, current_guess in enumerate(self.config_guess):
            # if chosen randomly, choose a random angle and scale based on wether it is the angle
            # of a problem hamiltonian sub circuit or mixing hamiltonian sub circuit
            if current_guess == "rand":
                next_angle = 2 * math.pi * (0.5 - np.random.rand()) 
                if idx % 2 == 0:
                    next_angle *= self.param_problem_range
                else:
                    next_angle *= self.param_mixing_range
            # The angle is fixed for all repetitions
            else:
                next_angle = current_guess
            inital_angles.append(next_angle)
        return np.array(inital_angles)

    

class QaoaParameterSupervisorGridSearch(QaoaParameterSupervisor):
    """a class for choosing qaoa parameters by searching using a grid on the given parameter space
    """

    def __init__(self, **args):
        """
        first calculates the grid that is going to be searched and then sets up the data structures
        necessary to keep track which grid points have already been tried output

        A grid is a product of grids on a each angle space of one layer of problem+mixing hamiltonian.
        We can give a default grid that each layer will use if their grid config doesn' specifiy it.
        A grid for one layers needs upper and lower bounds for the angle of the mixing and problem
        hamiltonian. Furthermore, it either needs to specifiy how many grid points are added for each angle
        Keep in mind that the total number of grids point is the product over all grids over all layers
        which can grow very quickly.
        Each Grid is represented as a dictionary with 6 Values
            lowerBoundMixing, upperBoundMixing, numGridpointsMixing
            lowerBoundProblem, upperBoundProblem, numGridpointsProblem
    
        Args:
            defaultGrid: (dict) the default paramters of the grid for one layer
                keys
                    lowerBoundProblem :
                    upperBoundProblem :
                    numGridpointsProblem :
                    lowerBoundMixing :
                    upperBoundMixing :
                    numGridpointsMixing :
            gridList: list
                a list of dictionaries. The i-th dicitonary contains the grid values of the i-th layer
        """
        self.setDefaultGrid(args.get('defaultGrid',{}))
        self.config_qaoa = args['qaoaOptimizer'].config_qaoa
        self.gridsByLayer = [grid for layer in self.config_qaoa['initial_guess'] 
                                  for grid in self.transformToGridpoints(layer)]
        self.currentGuessIndex = -1
        self.intialGuessList = product(*self.gridsByLayer)
        self.num_vars = 2 * len(self.config_qaoa['initial_guess'])


    def getNumAngles(self):
        return self.num_vars

    
    def getInitialAngleIterator(self):
        for angleList in product(*self.gridsByLayer):
            yield np.array(angleList)
            

    def setDefaultGrid(self, defaultGrid: dict):
        """
        reads a grid dictionary and saves them to be accessible later as a default fallback value 
        in case they aren't specified for one a layer of qaoa
    
        Args:
            defaultGrid: (dict) 
                a dictionary with values to specify a grid. For this grid, there also
                exist default values to be used as default values
        Returns:
            (None) modifies the attribute self.default
        """
        self.default = {
                    "lowerBoundProblem" : defaultGrid.get("lowerBoundProblem" , - math.pi ),
                    "upperBoundProblem" :defaultGrid.get("upperBoundProblem" , math.pi),
                    "numGridpointsProblem" : defaultGrid.get("numGridpointsProblem" , 3),
                    "lowerBoundMixing" : defaultGrid.get("lowerBoundMixing" , -math.pi),
                    "upperBoundMixing" : defaultGrid.get("upperBoundMixing" , math.pi),
                    "numGridpointsMixing" :defaultGrid.get("numGridpointsMixing" , 3),
        }
    

    def transformToGridpoints(self, gridDict):
        """
        returns two list of grid points based on the dictionary describing the grid points of one layer.
    
        Args:
            gridDict: (dict) a dicitonary with the following keys
                    lowerBoundProblem 
                    upperBoundProblem 
                    numGridpointsProblem 
                    lowerBoundMixing 
                    upperBoundMixing 
                    numGridpointsMixing 
        Returns:
            (list, list) returns two lists with float values
        """
        problemGrid = self.makeGridList(
                lowerBound = gridDict.get('lowerBoundProblem', self.default['lowerBoundProblem']),
                upperBound = gridDict.get('upperBoundProblem', self.default['upperBoundProblem']),
                numGridpoints = gridDict.get('numGridpointsProblem', self.default['numGridpointsProblem']),
        )
        mixingGrid = self.makeGridList(
                lowerBound = gridDict.get('lowerBoundMixing', self.default['lowerBoundMixing']),
                upperBound = gridDict.get('upperBoundMixing', self.default['upperBoundMixing']),
                numGridpoints = gridDict.get('numGridpointsMixing', self.default['numGridpointsMixing']),
        )
        return problemGrid, mixingGrid

    def makeGridList(self, lowerBound, upperBound, numGridpoints):
        """
        takes lower and upper bound and returns a list of equidistant points in that interval
        with length equal the specified grid points. 
    
        Args:
            lowerBound: (float) the minimal grid point
            upperBound: (float) the maximal grid point
            numGridpoints: (int) the total number of grid points
        Returns:
            (list) a list of floating values which form a grid on the given intervall
        """
        if numGridpoints == 0:
            raise ValueError("trying to construct an empty grid set, which vanishes in the product of grids")
        stepSize = float(upperBound - lowerBound) / (numGridpoints-1)
        return [lowerBound + idx * stepSize for idx in range(numGridpoints)]
    

    def chooseInitalParameters(self):
        res =  np.array(next(self.intialGuessList))
        return res

    


class QaoaQiskit(BackendBase):
    def __init__(self, reader: InputReader):
        super().__init__(reader)

        # copy relevant config to make code more readable
        self.config_qaoa = self.config["QaoaBackend"]
        self.addResultsDict()
        self.parameterSupervisior = QaoaParameterSupervisor.makeParameterSupervisior(
                    supervisior_type = self.config_qaoa.get('supervisior_type', 'RandomOrFixed'),
                    qaoaOptimizer = self
        )
        self.numAngles = self.parameterSupervisior.getNumAngles()

        # initiate local parameters
        self.iterationCounter = None
        self.iter_result = {}
        self.rep_result = {}
        self.qc = None
        self.paramVector = None
        self.statistics = {"confidence": 0.0,
                           "bestBitstring": "",
                           "probabilities": {},
                           "pValues": {},
                           "uValues": {}
                           }

        # set up connection to IBMQ servers
        if self.config_qaoa["noise"] or (not self.config_qaoa["simulate"]):
            IBMQ.save_account(self.config["APItoken"]["IBMQ_API_token"], overwrite=True)
            self.provider = IBMQ.load_account()

    def transformProblemForOptimizer(self):
        """
        Initializes an IsingInterface-instance, which encodes the Ising Spin Glass Problem, using the network to be
        optimized.

        Returns:
            (IsingBackbone) The IsingInterface-instance, which encodes the Ising Spin Glass Problem.
        """
        self.transformedProblem = IsingBackbone.buildIsingProblem(
            network=self.network, config=self.config["IsingInterface"]
        )
        self.output["results"]["qubit_map"] = self.transformedProblem.getQubitMapping()

    def optimize(self):
        """
        Optimizes the network encoded in the IsingInterface-instance. A self-written Qaoa algorithm is used, which can
        either simulate the quantum part or solve it on one of IBMQ's servers (provided the correct credentials).
        As classic solvers SPSA, COBYLA or ADAM can be chosen.

        Returns:
            (None) The optimized solution is stored in the self.output dictionary.
        """
        # retrieve various parameters from the config
        shots = self.config_qaoa["shots"]
        simulator = self.config_qaoa["simulator"]
        simulate = self.config_qaoa["simulate"]
        noise = self.config_qaoa["noise"]
        initial_guess_original = np.array([0 for i in range(self.parameterSupervisior.getNumAngles())])
        num_vars = self.numAngles
        max_iter = self.config_qaoa["max_iter"]
        repetitions = self.config_qaoa["repetitions"]
        totalRepetition = 0


        hamiltonian = self.transformedProblem.getHamiltonianMatrix()
        scaledHamiltonian = self.scaleHamiltonian(hamiltonian=hamiltonian)
        self.output["results"]["hamiltonian"]["original"] = hamiltonian
        self.output["results"]["hamiltonian"]["scaled"] = scaledHamiltonian
        nqubits = len(hamiltonian)

        # create ParameterVector to be used as placeholder when creating the quantum circuit
        self.paramVector = ParameterVector("theta", self.numAngles)
        self.qc = self.create_qc(hamiltonian=scaledHamiltonian, theta=self.paramVector)
        # bind variables beta and gamma to qc, to generate a circuit which is saved in output as latex source code.
        drawTheta = self.createDrawTheta(theta=initial_guess_original)
        print(drawTheta)
        qcDraw = self.qc.bind_parameters({self.paramVector: drawTheta})
        self.output["results"]["qc"] = qcDraw.draw(output="latex_source")

        # setup IBMQ backend and save its configuration to output
        backend, noise_model, coupling_map, basis_gates = self.setup_backend(
            simulator=simulator, simulate=simulate, noise=noise, nqubits=nqubits
        )
        self.output["results"]["backend"] = backend.configuration().to_dict()

        curRepetition = 1
        for initial_guess in self.parameterSupervisior.getInitialAngleIterator():
            time_start = datetime.timestamp(datetime.now())
            totalRepetition = curRepetition
            print(
                f"----------------------- Repetition {totalRepetition} ----------------------------------"
            )


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
            self.rep_result["optimizedResult"] = {
                "x": list(res[0]),  # solution [beta, gamma]
                "fun": res[1],  # objective function value
                "counts": self.rep_result["iterations"][res[2]]["counts"],  # counts of the optimized result
                "nfev": res[2],  # number of objective function calls
            }

            time_end = datetime.timestamp(datetime.now())
            duration = time_end - time_start
            self.rep_result["duration"] = duration

            self.output["results"]["repetitions"][totalRepetition] = self.rep_result

            curRepetition += 1

        self.output["results"]["totalReps"] = totalRepetition

    def processSolution(self) -> None:
        """
        Post processing of the solution. Adds the components from the IsingInterface-instance to the output.

        Returns:
            (None) The self.output dictionary is modified with post-process information.
        """
        self.output["components"] = self.transformedProblem.getData()

        self.extractPvalues()  # get probabilities of bitstrings
        self.findBestBitstring()
        self.compareBitStringToRest()  # one-sided Mann-Witney U Test
        self.determineConfidence()  # check p against various alphas

        self.output["results"]["statistics"] = self.statistics

        self.writeReportToOutput(bestBitstring=self.statistics["bestBitstring"])

    def transformSolutionToNetwork(self) -> None:
        """
        Encodes the optimal solution found during optimization and stored in self.output into a pypsa.Network. It reads
        the solution stored in the optimizer instance, prints some information about it and then writes it to the
        network.

        Returns:
            (None) Stores the outputNetwork as dicitonary in the self.output dictionary
        """
        self.printReport()
        bestBitstring = self.output["results"]["statistics"]["bestBitstring"]
        solution = []
        for idx, bit in enumerate(bestBitstring):
            if bit == "1":
                solution.append(idx)
        outputNetwork = self.transformedProblem.setOutputNetwork(solution=solution)
        outputDataset = outputNetwork.export_to_netcdf()
        self.output["network"] = outputDataset.to_dict()

    def addResultsDict(self) -> None:
        """
        Adds the basic structure for the self.output["results"]-dictionary.

        Returns:
            (None) Modifies the self.output["results"]-dictionary.
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
            "totalReps": 0,
            "repetitions": {},
        }

    def prepareRepetitionDict(self) -> None:
        """
        Initializes the basic structure for the self.rep_result-dictionary, setting its values to empty dictionaries,
        empty lists or None values.

        Returns:
            (None) Modifies the self.rep_result-dictionary.
        """
        self.rep_result = {
            "initial_guess": [],
            "duration": None,
            "optimizedResult": {},
            "iterations": {},
        }

    def prepareIterationDict(self) -> None:
        """
        Initializes the basic structure for the self.iter_result-dictionary, setting its values to empty dictionaries,
        empty lists or None values.

        Returns:
            (None) Modifies the self.iter_result-dictionary.
        """
        self.iter_result = {"theta": [], "counts": {}, "bitstrings": {}, "return": None}

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
            drawTheta.append(Parameter(f"\u03B2{layer + 1}"))  # append beta_i
            drawTheta.append(Parameter(f"\u03B3{layer + 1}"))  # append gamma_i
            # drawTheta.append(f"{chr(946)}{chr(8320 + layer)}")  #append beta_i
            # drawTheta.append(f"{chr(947)}{chr(8320 + layer)}")  #append gamma_i

        return drawTheta

    def getClassicalOptimizer(self, max_iter: int) -> qiskit.algorithms.optimizers:
        """
        Initiates and returns the classical optimizer set in the config file. If another optimizer than SPSA, COBYLA or
        Adam is specified, an error is thrown.

        Args:
            max_iter: (int) The maximum number of iterations the classical optimizer is allowed to perform.

        Returns:
            (qiskit.algorithms.optimizers) The initialized classical optimizer, as specified in the config.
        """
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

    def getMinCFvars(self) -> list:
        """
        Searches through the optimization results of all repetitions done with random (or partly random) initial values
        for beta and gamma and finds the minimum cost function value. The associated beta and gamma values will be
        returned as a list to be used in the refinement phase of the optimization.

        Returns:
            (list) The beta and gamma values associated with the minimal cost function value.
        """
        searchData = self.output["results"]["repetitions"]
        minCF = searchData[1]["optimizedResult"]["fun"]
        minX = []
        for i in range(1, len(searchData) + 1):
            if searchData[i]["optimizedResult"]["fun"] <= minCF:
                minCF = searchData[i]["optimizedResult"]["fun"]
                minX = searchData[i]["optimizedResult"]["x"]

        return minX

    def scaleHamiltonian(self, hamiltonian: list) -> list:
        """
        Scales the hamiltonian so that the maximum absolute value in the input hamiltonian is equal to Pi.

        Args:
            hamiltonian: (list) The input hamiltonian to be scaled.

        Returns:
            (list) The scaled hamiltonian.
        """
        matrixMax = np.max(hamiltonian)
        matrixMin = np.min(hamiltonian)
        matrixExtreme = max(abs(matrixMax), abs(matrixMin))
        factor = matrixExtreme / math.pi
        scaledHamiltonian = np.array(hamiltonian) / factor

        return scaledHamiltonian.tolist()

    def create_qc(self, hamiltonian: list, theta: ParameterVector) -> QuantumCircuit:
        """
        Creates a qiskit quantum circuit based on the hamiltonian matrix given. The quantum circuit will be created
        using a ParameterVector to create placeholders, which can be filled with the actual parameters using qiskit's
        bind_parameters function.

        Args:
            hamiltonian: (dict) The matrix representing the problem Hamiltonian.
            theta: (ParameterVector) The ParameterVector of the same length as the list of optimizable parameters.

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
            qc.barrier()  # for visual purposes only, when the quantum circuit is drawn
            qc.barrier()
            # add problem Hamiltonian
            for i in range(len(hamiltonian)):
                for j in range(i, len(hamiltonian[i])):
                    if hamiltonian[i][j] != 0.0:
                        if i == j:
                            qc.rz(
                                -hamiltonian[i][j] * gammaValues[layer], i
                            )  # inversed, as the implementation in the IsingInterface inverses the values
                        else:
                            qc.rzz(
                                -hamiltonian[i][j] * gammaValues[layer], i, j
                            )  # inversed, as the implementation in the IsingInterface inverses the values
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
            bitstring: (str) The bitstring encoding a possible solution to the network.

        Returns:
            (float) The absolut deviation from the optimal (0), where the kirchhoff constrains would be completely
                    satisfied for the given network.
        """
        try:
            # check if the kirchhoff costs for this bitstring have already been calculated and if so use this value
            return self.output["results"]["kirchhoff"][bitstring]["total"]
        except KeyError:
            self.output["results"]["kirchhoff"][bitstring] = {}
            kirchhoffCost = 0.0
            # calculate the deviation from the optimal for each bus separately
            for bus in self.network.buses.index:
                bitstringToSolution = [
                    idx for idx, bit in enumerate(bitstring) if bit == "1"
                ]
                for _, val in self.transformedProblem.calcPowerImbalanceAtBus(
                        bus, bitstringToSolution
                ).items():
                    # store the penalty for each bus and then add them to the total costs
                    self.output["results"]["kirchhoff"][bitstring][bus] = val
                    kirchhoffCost += abs(val) ** 2
            self.output["results"]["kirchhoff"][bitstring]["total"] = kirchhoffCost
            return kirchhoffCost

    def compute_expectation(self, counts: dict) -> float:
        """
        Computes expectation values based on the measurement/simulation results.

        Args:
            counts: (dict) The number of times as specific bitstring was measured. The bitstring is the key and its
            count the value.

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

    def setup_backend(self, simulator: str, simulate: bool, noise: bool, nqubits: int
                      ) -> Tuple[BaseBackend, NoiseModel, list, list]:
        """
        Sets up the qiskit backend based on the settings passed into the function.

        Args:
            simulator: (str) The name of the Quantum Simulator to be used, if simulate is True.
            simulate: (bool) If True, the specified Quantum Simulator will be used to execute the Quantum Circuit.
                             If False, the least busy IBMQ Quantum Comupter will be initiated to be used to execute the
                             Quantum Circuit.
            noise: (bool) If True, noise will be added to the Simulator. If False, no noise will be added. Only works
                          if "simulate" is set to True. On actual IBMQ devices noise is always present and cannot be
                          deactivated
            nqubits: (int) The number of Qubits of the Quantum Circuit. Used to find a suitable IBMQ Quantum Computer.

        Returns:
            (BaseBackend) The backend to be used.
            (NoiseModel) The noise model of the chosen backend, if noise is set to True. Otherwise it is set to None.
            (list) The coupling map of the chosen backend, if noise is set to True. Otherwise it is set to None.
            (list) The initial basis gates used to compile the noise model, if noise is set to True. Otherwise it is set
                   to None.
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
                filters=lambda x: x.configuration().n_qubits > nqubits
                                  and not x.configuration().simulator
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
    ):
        """
        Builds the objective function, which can be used in a classical solver.

        Args:
            backend: (BaseBackend) The backend to be used.
            noise_model: (NoiseModel) The noise model of the chosen backend. Default: None
            coupling_map: (list) The coupling map of the chosen backend Default: None
            basis_gates: (list) The initial basis gates used to compile the noise model Default: None
            shots: (int) The number of repetitions of each circuit, for sampling. Default: 1024
            simulate: (bool) If True, the specified Quantum Simulator will be used to execute the Quantum Circuit. If
                             False, the IBMQ Quantum Comupter set in setup_backend will be used to execute the Quantum
                             Circuit. Default: True

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

    def extractPvalues(self) -> None:
        """
        Searches through the results and combines the probability for each bitstring in each repetition of the
        "optimizedResult" in lists. E.g. For 100 repetitions, a list for each bitstring is created, containing the 100
        probability values (one from each repetition)

        Returns:
            (None) Writes the created lists of probabilities into the self.statistics dictionary.
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
                if bitstring in data[key]["optimizedResult"]["counts"]:
                    p = data[key]["optimizedResult"]["counts"][bitstring] / shots
                else:
                    p = 0
                probabilities[bitstring].append(p)
        self.statistics["probabilities"] = probabilities

    def findBestBitstring(self) -> None:
        """
        Takes the median of the probabilites of each bitstring and determines, which bitstring has objectively the
        highest probability and stores this bitstring in self.statistics.

        Returns:
            (None) Modifies the self.statistics dictionary.
        """
        probabilities = self.statistics["probabilities"]
        bestBitstring = list(probabilities.keys())[0]  # set first bitstring as best for now
        bestMedian = median(probabilities[bestBitstring])  # get median of first bitstring

        for bitstring in probabilities:
            if median(probabilities[bitstring]) > bestMedian:
                bestMedian = median(probabilities[bitstring])
                bestBitstring = bitstring
        self.statistics["bestBitstring"] = bestBitstring

    def compareBitStringToRest(self) -> None:
        """
        Compares the bestBitstring (found in the findBestBitstring function) to every other bitstring using a one-
        sided Mann Whitney U Test, where the alternative hypothesis is that the probability to find the bestBitstring is
        greater than the probabilities of the other bitstrings. The results of the tests are stored in the
        self.statistics dictionary.

        Returns:
            (None) Modifies the self.statistics dictionary.
        """
        bestBitstring = self.statistics["bestBitstring"]
        probabilities = self.statistics["probabilities"]
        for bitstring in probabilities.keys():
            if bitstring == bestBitstring:
                continue
            u, p = stats.mannwhitneyu(x=probabilities[bestBitstring],
                                      y=probabilities[bitstring],
                                      alternative="greater")
            self.statistics["pValues"][f"{bestBitstring}-{bitstring}"] = float(p)
            self.statistics["uValues"][f"{bestBitstring}-{bitstring}"] = float(u)

    def determineConfidence(self) -> None:
        """
        Determines with which confidence, if any, the bestBitstring can be found. A list of alphas is checked, starting
        at 0.01 up until 0.5. The found confidence is then stored in self.statistics. If none is found the value in
        self.statistics["confidence"] is kept at 0.0, thus indicating no bestBitstring can be confidently determined.

        Returns:
            (None) Modifies the self.statistics dictionary.
        """
        alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        for alpha in alphas:
            broken = False
            for key, value in self.statistics["pValues"].items():
                if value > alpha:
                    broken = True
                    break
            if not broken:
                self.statistics["confidence"] = 1 - alpha
                break

    def writeReportToOutput(self, bestBitstring: str):
        """
        Writes solution specific values of the optimizer result and the ising spin glass problem solution to the output
        dictionary.

        Args:
            bestBitstring: (str) The bitstring representing the best solution found during optimization.

        Returns:
            (None) Modifies self.output with solution specific parameters and values.
        """
        solution = []
        for idx, bit in enumerate(bestBitstring):
            if bit == "1":
                solution.append(idx)
        report = self.transformedProblem.generateReport(solution=solution)
        for key in report:
            self.output["results"][key] = report[key]
