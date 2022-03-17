import copy
import json, yaml
import math

import numpy as np
import pypsa
import os.path

from numpy import random

from .BackendBase import BackendBase                                # import for Docker run
from .IsingPypsaInterface import IsingPypsaInterface                # import for Docker run
#from BackendBase import BackendBase                                # import for local/debug run
#from IsingPypsaInterface import IsingPypsaInterface                # import for local/debug run
#from EnvironmentVariableManager import EnvironmentVariableManager  # import for local/debug run
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit import Parameter


class QaoaQiskit(BackendBase):
    def __init__(self, config: dict,):
        super().__init__(config=config)
        self.metaInfo["results"] = {}
        self.resetResultDict()

        self.kirchhoff = {}
        self.components = {}
        self.docker = os.environ.get("RUNNING_IN_DOCKER", False)
        if config["QaoaBackend"]["noise"] or (not config["QaoaBackend"]["simulate"]):
            IBMQ.save_account(config["APItoken"]["IBMQ_API_token"], overwrite=True)
            self.provider = IBMQ.load_account()

    def resetResultDict(self):
        self.results_dict = {"iter_count": 0,
                             "simulate": None,
                             "noise": None,
                             "backend_name": None,
                             "shots": None,
                             "components": {},
                             "qc": None,
                             "initial_guess": [],
                             "duration": None,
                             "optimizeResults": {},
                             }

    def transformProblemForOptimizer(self, network):
        print("transforming problem...")
        return self.getComponents(network=network)

    def transformSolutionToNetwork(self, network, transformedProblem, solution):
        return network

    def processSolution(self, network, transformedProblem, solution):
        return solution

    def optimize(self, transformedProblem):

        shots = self.config["QaoaBackend"]["shots"]
        simulator = self.config["QaoaBackend"]["simulator"]
        simulate = self.config["QaoaBackend"]["simulate"]
        noise = self.config["QaoaBackend"]["noise"]
        initial_guess_original = copy.deepcopy(self.config["QaoaBackend"]["initial_guess"])
        num_vars = len(initial_guess_original)
        max_iter = self.config["QaoaBackend"]["max_iter"]
        repetitions = self.config["QaoaBackend"]["repetitions"]
        repetitions_start = 1

        if "rand" in initial_guess_original:
            randRep = 2
        else:
            randRep = 1

        for rand in range(randRep):
            for i in range(1, repetitions + 1):
                time_start = datetime.timestamp(datetime.now())
                print(f"----------------------- Iteration {rand * repetitions + i} ----------------------------------")

                initial_guess = []

                for j in range(num_vars):
                    # choose initial guess randomly (between 0 and 2PI for beta and 0 and PI for gamma)
                    if initial_guess_original[j] == "rand":
                        if j % 2 == 0:
                            initial_guess.append((0.5 - random.rand()) * 2 * math.pi)
                        else:
                            initial_guess.append((0.5 - random.rand()) * 2 * math.pi)
                    else:
                        initial_guess.append(initial_guess_original[j])
                initial_guess = np.array(initial_guess)

                self.resetResultDict()
                self.results_dict["initial_guess"] = initial_guess.tolist()
                self.results_dict["components"] = self.components


                filename_input = str(self.config["QaoaBackend"]["filenameSplit"][1]) + "_" + \
                                 str(self.config["QaoaBackend"]["filenameSplit"][2]) + "_" + \
                                 str(self.config["QaoaBackend"]["filenameSplit"][3]) + "_" + \
                                 str(self.config["QaoaBackend"]["filenameSplit"][4])
                filename_date = self.config["QaoaBackend"]["filenameSplit"][7]
                filename_time = self.config["QaoaBackend"]["filenameSplit"][8]
                filename_config = str(self.config["QaoaBackend"]["filenameSplit"][9])
                if len(self.config["QaoaBackend"]["filenameSplit"]) == 11:
                    filename_config += "_" + str(self.config["QaoaBackend"]["filenameSplit"][10])

                filename = f"Qaoa_{filename_input}_{filename_date}_{filename_time}_{filename_config}__" \
                           f"{rand * repetitions + i}.json"

                expectation = self.get_expectation(filename=filename,
                                                   components=transformedProblem,
                                                   simulator=simulator,
                                                   shots=shots,
                                                   simulate=simulate,
                                                   noise=noise)

                if self.config["QaoaBackend"]["classical_optimizer"] == "SPSA":
                    optimizer = SPSA(maxiter=max_iter, blocking=False)
                elif self.config["QaoaBackend"]["classical_optimizer"] == "COBYLA":
                    optimizer = COBYLA(maxiter=max_iter, tol=0.0001)
                else:
                    optimizer = COBYLA(maxiter=max_iter, tol=0.0001)

                res = optimizer.optimize(num_vars=num_vars, objective_function=expectation, initial_point=initial_guess)
                self.results_dict["optimizeResults"]["x"] = list(res[0])  # solution [beta, gamma]
                self.results_dict["optimizeResults"]["fun"] = res[1]  # objective function value
                self.results_dict["optimizeResults"]["nfev"] = res[2]  # number of objective function calls

                #res = optimizer.minimize(fun=expectation, x0=initial_guess)
                #self.results_dict["optimizeResults"]["x"] = list(res.x)  # solution [beta, gamma]
                #self.results_dict["optimizeResults"]["fun"] = float(res.fun)  # objective function value
                #self.results_dict["optimizeResults"]["nfev"] = int(res.nfev)  # number of objective function calls

                time_end = datetime.timestamp(datetime.now())
                duration = time_end - time_start
                self.results_dict["duration"] = duration

                # safe final results
                if self.docker:
                    with open(f"Problemset/{filename}", "w") as write_file:
                        json.dump(self.results_dict, write_file, indent=2, default=str)
                else:
                    with open(os.path.dirname(__file__) + "/../../sweepNetworks/" + filename, "w") as write_file:
                        json.dump(self.results_dict, write_file, indent=2, default=str)

                last_rep = self.results_dict["iter_count"]
                last_rep_counts = self.results_dict[f"rep{last_rep}"]["counts"]
                self.metaInfo["results"][i] = {"filename": filename,
                                               "backend_name": self.results_dict["backend_name"],
                                               "initial_guess": self.results_dict["initial_guess"],
                                               "optimize_Iterations": self.results_dict["iter_count"],
                                               "optimizeResults": self.results_dict["optimizeResults"],
                                               "duration": duration,
                                               "counts": last_rep_counts}

            if "rand" in initial_guess_original:
                randFileName = ""
                for namePart in self.config["QaoaBackend"]["filenameSplit"]:
                    randFileName += namePart + "_"
                randFileName += "rand"

                if self.docker:
                    with open(f"Problemset/{randFileName}", "w") as write_file:
                        json.dump(self.metaInfo, write_file, indent=2, default=str)
                else:
                    with open(os.path.dirname(__file__) + "/../../sweepNetworks/" + randFileName, "w") as write_file:
                        json.dump(self.metaInfo, write_file, indent=2, default=str)

                #repetitions_start += repetitions
                #repetitions += repetitions
                minCFvars  = self.getMinCFvars()
                for j in range(num_vars):
                    if initial_guess_original[j] == "rand":
                        initial_guess_original[j] = minCFvars[j]
                self.metaInfo["results"] = {}


        self.metaInfo["kirchhoff"] = self.kirchhoff[f"rep{last_rep}"]

        return self.metaInfo["results"]

    def getMinCFvars(self):
        """
        Searches through metaInfo["results"] and finds the minimum cost function value along with the associated betas
        and gammas, which will be returned as a list.
        Returns:
            minX (list) a list with the betas and gammas associated with the minimal cost function value.
        """
        minCF = self.metaInfo["results"][1]["optimizeResults"]["fun"]
        minX = []
        for i in range(1, len(self.metaInfo["results"]) + 1):
            if self.metaInfo["results"][i]["optimizeResults"]["fun"] <= minCF:
                minCF = self.metaInfo["results"][i]["optimizeResults"]["fun"]
                minX = self.metaInfo["results"][i]["optimizeResults"]["x"]

        return minX

    def getMetaInfo(self):
        return self.metaInfo

    def validateInput(self, path, network):
        pass

    def handleOptimizationStop(self, path, network):
        pass

    def get_power(self, comp: str, network: pypsa.Network, type: str) -> float:
        """
        Extracts the power value of the given component and adjusts its sign according to the function this component
        fulfills for the given bus.

        Args:
            comp: (str) The component from which the power should be extracted.
            network: (pypsa.Network) The PyPSA network to be analyzed.
            type: (str) The type of the component, "line" or "gen".

        Returns:
            (float) The power value for the given comp.
        """
        if type == "gen":
            return float(network.generators[network.generators.index == comp].p_nom)
        elif type == "line":
            return float(network.lines[network.lines.index == comp].s_nom)

    def buildPowerAndQubits(self, components: dict, network: pypsa.Network, bus: str) -> dict:
        """
        Builds up the power and qubits lists stored in the components dictionary and returns it.

        Args:
            components: (dict) All components to be modeled as a Quantum Circuit.
            network: (pypsa.Network) The PyPSA network to be analyzed.
            bus: (str) The bus where the components are connected.

        Returns:
            (dict) The components dictionary with the power and qubits lists for this bus.
        """
        qubit_map = components["qubit_map"]
        for comp in components[bus]["generators"]:
            components[bus]["qubits"].append(qubit_map[comp][0])
            components[bus]["power"].append(self.get_power(comp=comp, network=network, type="gen"))
        for comp in components[bus]["positiveLines"]:
            components[bus]["power"].append(self.get_power(comp=comp, network=network, type="line"))
            if network.lines[network.lines.index == comp].bus1[0] == bus:
                components[bus]["qubits"].append(qubit_map[comp][0])
            elif network.lines[network.lines.index == comp].bus0[0] == bus:
                components[bus]["qubits"].append(qubit_map[comp][1])
        for comp in components[bus]["negativeLines"]:
            components[bus]["power"].append(-self.get_power(comp=comp, network=network, type="line"))
            if network.lines[network.lines.index == comp].bus1[0] == bus:
                components[bus]["qubits"].append(qubit_map[comp][1])
            elif network.lines[network.lines.index == comp].bus0[0] == bus:
                components[bus]["qubits"].append(qubit_map[comp][0])

        return components

    def getComponents(self, network: pypsa.Network) -> dict:
        """
        Separates and organizes all components of a network to be optimized using QAOA.
        For each bus the generators, positive lines (which add power to the bus), negative lines (which remove power
        from the bus) and load are accessed and stored in the dictionary, together with a flattened list of generators,
        positive and negative lines, and the power associated to these components. At last the components are mapped
        on logical qubits.

        Args:
            network: (pypsa.Network) The PyPSA network to be analyzed.

        Returns:
            (dict) All components to be modeled as a Quantum Circuit.
        """
        components = {}

        if self.config["QaoaBackend"]["qcGeneration"] == "Iteration" or \
                self.config["QaoaBackend"]["qcGeneration"] == "IterationMatrix":
            qubit_map = {}
            qubit = 0
            for comp in list(network.generators.index):
                qubit_map[comp] = [qubit]
                qubit += 1
            for comp in list(network.lines.index):
                qubit_map[comp] = [qubit, qubit + 1]  # [line going to bus1, line going to bus0]
                qubit += 2
        elif self.config["QaoaBackend"]["qcGeneration"] == "Ising":
            transformedProblem = IsingPypsaInterface.buildCostFunction(network=network)
            qubit_map = transformedProblem.getQubitMapping()

        components["qubit_map"] = qubit_map

        for bus in network.buses.index.values:
            components[bus] = {"generators": list(network.generators[network.generators.bus == bus].index),
                               "positiveLines": list(network.lines[network.lines.bus1 == bus].index) +
                                                list(network.lines[network.lines.bus0 == bus].index),
                               # lines are bidirectional
                               "negativeLines": list(network.lines[network.lines.bus1 == bus].index) +
                                                list(network.lines[network.lines.bus0 == bus].index),
                               "load": sum(list(network.loads[network.loads.bus == bus].p_set)), }
            components[bus][f"flattened_{bus}"] = components[bus]["generators"] + \
                                                  components[bus]["positiveLines"] + \
                                                  components[bus]["negativeLines"]
            components[bus]["power"] = []
            components[bus]["qubits"] = []
            components = self.buildPowerAndQubits(components=components, network=network, bus=bus)

        if self.config["QaoaBackend"]["qcGeneration"] == "IterationMatrix":
            lastKey = list(components["qubit_map"])[-1]
            nqubits = components["qubit_map"][lastKey][-1] + 1
            hamiltonian = self.calculateHpMatrix(components=components, nqubits=nqubits)
        elif self.config["QaoaBackend"]["qcGeneration"] == "Ising":
            hamiltonian = transformedProblem.getHamiltonianMatrix()
        else:
            hamiltonian = []
        scaledHamiltonian = self.scaleHamiltonian(hamiltonian=hamiltonian)

        components["hamiltonian"] = {}
        components["hamiltonian"]["scaled"] = scaledHamiltonian
        components["hamiltonian"]["original"] = hamiltonian

        self.components = components

        return components

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

    def addHpIter(self, qc: QuantumCircuit, gamma: float, components: dict) -> QuantumCircuit:
        """
        Appends the problem Hamiltonian to a quantum circuit by iterating through the cost function and appending
        single gates.

        Args:
            qc: (QuantumCircuit) The quantum circuit to be appended with the problem Hamiltonian.
            gamma: (float) The optimizable value for the problem Hamiltonian.
            components: (dict) All components to be modeled as a Quantum Circuit.

        Returns:
            (QuantumCircuit) The appended quantum circuit.
        """
        for bus in components:
            if bus != "qubit_map" and bus != "hamiltonian":
                length = len(components[bus][f"flattened_{bus}"])
                for i in range(length):
                    p_comp1 = components[bus]["power"][i]
                    qubit_comp1 = components[bus]["qubits"][i]
                    # negative load, since it removes power from the node
                    factor_load = -(components[bus]["load"]) * p_comp1
                    qc.rz(factor_load * gamma, qubit_comp1)
                    qc.barrier()
                    for j in range(length):
                        p_comp2 = components[bus]["power"][j]
                        qubit_comp2 = components[bus]["qubits"][j]
                        factor = 0.25 * p_comp1 * p_comp2
                        qc.rz(factor * gamma, qubit_comp1)
                        qc.rz(factor * gamma, qubit_comp2)
                        if i != j:
                            qc.rzz(factor * gamma, qubit_comp1, qubit_comp2)
                qc.barrier()
                qc.barrier()

        return qc

    def calculateHpMatrix(self, components: dict, nqubits: int) -> list:
        """
        Creates the Hamiltonian Matrix for the given problem.

        Args:
            components: (dict) All components to be modeled as a Quantum Circuit.
            nqubits: (int) The number of qubits in the network.

        Returns:
            (list) The Hamiltonian Matrix.
        """
        hp = [[0.0 for j in range(nqubits)] for i in range(nqubits)]
        for bus in components:
            if bus != "qubit_map" and bus != "hamiltonian":
                length = len(components[bus][f"flattened_{bus}"])
                for i in range(length):
                    p_comp1 = components[bus]["power"][i]
                    i_qubit = components[bus]["qubits"][i]
                    # negative load, since it removes power from the node
                    factor_load = -(components[bus]["load"]) * p_comp1
                    hp[i_qubit][i_qubit] += factor_load  # qc.rz(factor_load * gamma, qubit_comp1)
                    for j in range(length):
                        p_comp2 = components[bus]["power"][j]
                        j_qubit = components[bus]["qubits"][j]
                        factor = 0.25 * p_comp1 * p_comp2
                        hp[i_qubit][i_qubit] += factor  # qc.rz(factor * gamma, qubit_comp1)
                        hp[j_qubit][j_qubit] += factor  # qc.rz(factor * gamma, qubit_comp2)
                        if i != j:
                            hp[i_qubit][j_qubit] += factor  # qc.rzz(factor * gamma, qubit_comp1, qubit_comp2)
                            hp[j_qubit][i_qubit] += factor  # creates the symmetry in Hp
        return hp

    def addHpMatrix(self, qc: QuantumCircuit, gamma: float, nqubits: int, hp: list) -> QuantumCircuit:
        """
        Appends the problem Hamiltonian to a quantum circuit using the Hamiltonian matrix.

        Args:
            qc: (QuantumCircuit) The quantum circuit to be appended with the problem Hamiltonian.
            gamma: (float) The optimizable value for the problem Hamiltonian.
            nqubits: (int) The number of qubits in the network.
            hp: (list) The Hamiltonian Matrix.

        Returns:
            (QuantumCircuit) The appended quantum circuit.
        """

        for i in range(nqubits):
            for j in range(i, nqubits):
                if hp[i][j] != 0.0:
                    if i == j:
                        qc.rz(hp[i][j] * gamma, i)
                    else:
                        qc.rzz(hp[i][j] * gamma, i, j)
        qc.barrier()
        qc.barrier()

        return qc

    def addHb(self, qc: QuantumCircuit, beta: float, nqubits: int) -> QuantumCircuit:
        """
        Appends the mixer Hamiltonian to a quantum circuit.

        Args:
            qc: (QuantumCircuit) The quantum circuit to be appended with the problem Hamiltonian.
            beta: (float) The optimizable value for the mixer Hamiltonian.
            nqubits: (int) The number of qubits in the network.

        Returns:
            (QuantumCircuit) The appended quantum circuit.
        """

        for i in range(nqubits):
            qc.rx(beta, i)
        qc.barrier()
        qc.barrier()

        return qc

    def create_qc1(self, components: dict, theta: list, hamiltonianType: str = "scaled") -> QuantumCircuit:
        """
        Creates a quantum circuit based on the components given and the cost function:

        Args:
            components: (dict) All components to be modeled as a Quantum Circuit.
            theta: (list) The optimizable values of the quantum circuit. Two arguments needed: beta = theta[0] and
                          gamma = theta[1].
            hamiltonianType: (str) The type of the hamiltonian matrix to be used. Options can be "scaled" or "original".
                                   Default: "scaled"

        Returns:
            (QuantumCircuit) The created quantum circuit.
        """
        lastKey = list(components["qubit_map"])[-1]
        nqubits = components["qubit_map"][lastKey][-1] + 1
        qc = QuantumCircuit(nqubits)

        beta = theta[0]
        gamma = theta[1]

        # add Hadamard gate to each qubit
        for i in range(nqubits):
            qc.h(i)
        qc.barrier()
        qc.barrier()

        if self.config["QaoaBackend"]["qcGeneration"] == "IterationMatrix":
            qc = self.addHpMatrix(qc=qc, gamma=gamma, nqubits=nqubits, hp=components["hamiltonian"][hamiltonianType])
        elif self.config["QaoaBackend"]["qcGeneration"] == "Iteration":
            qc = self.addHpIter(qc=qc, gamma=gamma, components=components)

        qc = self.addHb(qc=qc, beta=beta, nqubits=nqubits)

        qc.measure_all()

        return qc

    def create_qcIsing(self, hamiltonian: dict, theta: list) -> QuantumCircuit:
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

        beta = theta[0]
        gamma = theta[1]

        # add Hadamard gate to each qubit
        for i in range(nqubits):
            qc.h(i)
        qc.barrier()
        qc.barrier()

        # add problem Hamiltonian
        for i in range(len(hamiltonian)):
            for j in range(i, len(hamiltonian[i])):
                if hamiltonian[i][j] != 0.0:
                    if i == j:
                        qc.rz(-hamiltonian[i][j] * gamma, i)  # negative because it´s the inverse of original QC
                    else:
                        qc.rzz(-hamiltonian[i][j] * gamma, i, j)  # negative because it´s the inverse of original QC
        qc.barrier()
        qc.barrier()

        # add mixing Hamiltonian to each qubit
        for i in range(nqubits):
            qc.rx(beta, i)

        qc.measure_all()

        return qc

    def create_qc3(self, components: dict, theta: list, hamiltonianType: str = "scaled") -> QuantumCircuit:
        # 2 betas & 2 gammas
        """
        Creates a quantum circuit based on the components given and the cost function:

        Args:
            components: (dict) All components to be modeled as a Quantum Circuit.
            theta: (list) The optimizable values of the quantum circuit. Four arguments needed: [beta0, gamma0, beta1,
                          gamma1].
            hamiltonianType: (str) The type of the hamiltonian matrix to be used. Options can be "scaled" or "original".
                                   Default: "scaled"

        Returns:
            (QuantumCircuit) The created quantum circuit.
        """
        lastKey = list(components["qubit_map"])[-1]
        nqubits = components["qubit_map"][lastKey][-1] + 1
        qc = QuantumCircuit(nqubits)
        beta0 = theta[0]
        gamma0 = theta[1]
        beta1 = theta[2]
        gamma1 = theta[3]

        # add Hadamard gate to each qubit
        for i in range(nqubits):
            qc.h(i)
        qc.barrier()

        qc = self.addHpMatrix(qc=qc, gamma=gamma0, nqubits=nqubits, hp=components["hamiltonian"][hamiltonianType])
        qc = self.addHb(qc=qc, beta=beta0, nqubits=nqubits)

        qc = self.addHpMatrix(qc=qc, gamma=gamma1, nqubits=nqubits, hp=components["hamiltonian"][hamiltonianType])
        qc = self.addHb(qc=qc, beta=beta1, nqubits=nqubits)

        qc.measure_all()

        return qc

    def kirchhoff_satisfied2(self, bitstring: str, components: dict) -> float:
        """
        Checks if the kirchhoff constraints are satisfied for the given solution.

        Args:
            bitstring: (str) The possible solution to the network.
            components: (dict) All components to be modeled as a Quantum Circuit.

        Returns:
            (float) The absolut deviation from the optimal (0), where the kirchhoff constrains would be completely
                    satisfied for the given network.
        """

        self.kirchhoff[f"rep{self.results_dict['iter_count']}"][bitstring] = {}
        power_total = 0
        for bus in components:
            power = 0
            if bus != "qubit_map" and bus != "hamiltonian":
                power -= components[bus]["load"]
                i = 0
                for comp in components[bus][f"flattened_{bus}"]:
                    #i = components[bus][f"flattened_{bus}"].index(comp)
                    i_bit = components[bus]["qubits"][i]
                    power += (components[bus]["power"][i] * float(bitstring[i_bit]))
                    i += 1
                self.kirchhoff[f"rep{self.results_dict['iter_count']}"][bitstring][bus] = power
            power_total += abs(power)
        self.kirchhoff[f"rep{self.results_dict['iter_count']}"][bitstring]["total"] = power_total

        return power_total

    def compute_expectation(self, counts: dict, components: dict, filename: str) -> float:
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
            obj = self.kirchhoff_satisfied2(bitstring=bitstring, components=components)
            avg += obj * count
            sum_count += count
            self.results_dict[f"rep{self.results_dict['iter_count']}"][bitstring] = {"count": count,
                                                                                     "obj": obj,
                                                                                     "avg": avg,
                                                                                     "sum_count": sum_count}

        self.results_dict[f"rep{self.results_dict['iter_count']}"]["return"] = avg / sum_count

        # safe results to make sure nothing is lost, even if the code or backend crashes
        if self.docker:
            with open(f"Problemset/{filename}", "w") as write_file:
                json.dump(self.results_dict, write_file, indent=2, default=str)
        else:
            with open(os.path.dirname(__file__) + "/../../sweepNetworks/" + filename, "w") as write_file:
                json.dump(self.results_dict, write_file, indent=2, default=str)

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

    def get_expectation(self, filename: str, components: dict, simulator: str = "aer_simulator",
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
        lastKey = list(components["qubit_map"])[-1]
        nqubits = components["qubit_map"][lastKey][-1] + 1

        backend, noise_model, coupling_map, basis_gates = self.setup_backend(simulator=simulator,
                                                                             simulate=simulate,
                                                                             noise=noise,
                                                                             nqubits=nqubits)

        self.metaInfo["qaoaBackend"] = backend.configuration().to_dict()

        self.results_dict["shots"] = shots
        self.results_dict["simulate"] = simulate
        self.results_dict["noise"] = noise
        self.results_dict["backend_name"] = self.metaInfo["qaoaBackend"]["backend_name"]

        def execute_circ(theta):
            thetaDraw = [Parameter("\u03B2\u2081"), Parameter("\u03B3\u2081")]
            if self.config["QaoaBackend"]["qcGeneration"] == "Iteration" or \
                    self.config["QaoaBackend"]["qcGeneration"] == "IterationMatrix":
                if len(self.config["QaoaBackend"]["initial_guess"]) == 2:
                    qc = self.create_qc1(components=components, theta=theta)
                    qcDraw = self.create_qc1(components=components, theta=thetaDraw)
                elif len(self.config["QaoaBackend"]["initial_guess"]) == 4:
                    qc = self.create_qc3(components=components, theta=theta)
                    thetaDraw.extend([Parameter("\u03B2\u2082"), Parameter("\u03B3\u2082")])
                    qcDraw = self.create_qc3(components=components, theta=thetaDraw)
            elif self.config["QaoaBackend"]["qcGeneration"] == "Ising":
                qc = self.create_qcIsing(hamiltonian=components["hamiltonian"]["scaled"], theta=theta)
                qcDraw = self.create_qcIsing(hamiltonian=components["hamiltonian"]["scaled"], theta=thetaDraw)
            self.results_dict["qc"] = qcDraw.draw(output="latex_source")
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
            self.results_dict["iter_count"] += 1
            self.results_dict[f"rep{self.results_dict['iter_count']}"] = {}
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["theta"] = list(theta)
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["counts"] = counts

            self.kirchhoff[f"rep{self.results_dict['iter_count']}"] = {}

            return self.compute_expectation(counts=counts, components=components, filename=filename)

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

    qaoa = QaoaQiskit(config=config)
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
