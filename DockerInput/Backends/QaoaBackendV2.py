import json
import pypsa
import os.path
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit import Parameter


class QaoaQiskit():
    def __init__(self):
        self.results_dict = {"iter_count": 0,
                             "simulate": None,
                             "noise": None,
                             "shots": None,
                             "components": {},
                             "qc": None,
                             "initial_guess": [],
                             "duration": None,
                             "optimizeResults": {},
                             }

        with open(os.path.dirname(__file__) + "/../APItoken.json") as json_file:
            self.APItoken = json.load(json_file)

        self.kirchhoff = {}

    def power_extraction(self, comp: str, components: dict, network: pypsa.Network, bus: str) -> float:
        """
        Extracts the power value of the given component and adjusts its sign according to function this component
        fulfills for the given bus.

        Args:
            comp: (str) The component from which the power should be extracted.
            components: (dict) All components to be modeled as a Quantum Circuit.
            network: (pypsa.Network) The PyPSA network to be analyzed.
            bus: (str) The bus in which relation the component should be evaluated.

        Returns:
            (float) The adjusted power value for the given comp.
        """
        if comp in components[bus]["generators"]:
            return float(network.generators[network.generators.index == comp].p_nom)
        elif comp in components[bus]["positiveLines"]:
            return float(network.lines[network.lines.index == comp].s_nom)
        elif comp in components[bus]["negativeLines"]:
            return -float(network.lines[network.lines.index == comp].s_nom)

    def extract_power_list(self, components: dict, network: pypsa.Network, bus: str) -> list:
        """
        Extracts the power values of the components connected to a bus and stored in the flattened_{bus} list from the
        PyPSA network and stores them in a list with the same indices as flattened_{bus}.

        Args:
            components: (dict) All components to be modeled as a Quantum Circuit.
            network: (pypsa.Network) The PyPSA network to be analyzed.
            bus: (str) The bus where the components are connected.

        Returns:
            (list) The power values of the components in flattened_{bus}
        """
        power_list = [0] * len(components[bus][f"flattened_{bus}"])
        for comp in components[bus][f"flattened_{bus}"]:
            i = components[bus][f"flattened_{bus}"].index(comp)
            power_list[i] = self.power_extraction(comp=comp, components=components, network=network, bus=bus)

        return power_list

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
        for bus in network.buses.index.values:
            components[bus] = {"generators": list(network.generators[network.generators.bus == bus].index),
                               "positiveLines": list(network.lines[network.lines.bus1 == bus].index),
                               "negativeLines": list(network.lines[network.lines.bus0 == bus].index),
                               "load": sum(list(network.loads[network.loads.bus == bus].p_set)), }
            components[bus][f"flattened_{bus}"] = components[bus]["generators"] + \
                                                  components[bus]["positiveLines"] + \
                                                  components[bus]["negativeLines"]
            components[bus]["power"] = self.extract_power_list(components=components, network=network, bus=bus)

        qubit_map = {}
        qubit = 0
        for bus in components:
            for comp in components[bus][f"flattened_{bus}"]:
                if comp not in qubit_map:
                    qubit_map[comp] = qubit
                    qubit += 1

        components["qubit_map"] = qubit_map

        self.results_dict["components"] = components

        return components

    def create_qc1(self, components: dict, theta: list) -> QuantumCircuit:
        """
        Creates a quantum circuit based on the components given and the cost function:

        Args:
            components: (dict) All components to be modeled as a Quantum Circuit.
            theta: (list) The optimizable values of the quantum circuit. Two arguments needed: beta = theta[0] and
                          gamma = theta[1].

        Returns:
            (QuantumCircuit) The created quantum circuit.
        """
        nqubits = len(components["qubit_map"])
        qc = QuantumCircuit(nqubits)

        beta = theta[0]
        gamma = theta[1]

        # add Hadamard gate to each qubit
        for i in range(nqubits):
            qc.h(i)
        qc.barrier()

        # add problem Hamiltonian for each bus
        for bus in components:
            if bus is not "qubit_map":
                length = len(components[bus][f"flattened_{bus}"])
                for i in range(length):
                    p_comp1 = components[bus]["power"][i]
                    # negative load, since it removes power form the node
                    factor_load = -(components[bus]["load"]) * p_comp1
                    qc.rz(factor_load * gamma, i)
                    qc.barrier()
                    for j in range(length):
                        p_comp2 = components[bus]["power"][j]
                        factor = 0.25 * p_comp1 * p_comp2
                        qc.rz(factor * gamma, i)
                        qc.rz(factor * gamma, j)
                        if i != j:
                            qc.rzz(factor * gamma, i, j)
                        qc.barrier()

        # add mixing Hamiltonian to each qubit
        for i in range(nqubits):
            qc.rx(beta, i)

        qc.measure_all()

        return qc

    def create_qc2(self, components: dict, theta: list) -> QuantumCircuit:
        # 1 beta & 2 gammas
        """
        Creates a quantum circuit based on the components given and the cost function:

        Args:
            components: (dict) All components to be modeled as a Quantum Circuit.
            theta: (list) The optimizable values of the quantum circuit. Two arguments needed: beta = theta[0] and
                          gamma = theta[1].

        Returns:
            (QuantumCircuit) The created quantum circuit.
        """
        nqubits = len(components["qubit_map"])
        qc = QuantumCircuit(nqubits)

        beta = theta[0]
        gamma = theta[1]

        # add Hadamard gate to each qubit
        for i in range(nqubits):
            qc.h(i)
        qc.barrier()

        # add problem Hamiltonian for each bus
        index = 0
        for bus in components:
            index += 1
            if bus is not "qubit_map":
                gamma = theta[index]
                length = len(components[bus][f"flattened_{bus}"])
                for i in range(length):
                    p_comp1 = components[bus]["power"][i]
                    # negative load, since it removes power form the node
                    factor_load = -(components[bus]["load"]) * p_comp1
                    qc.rz(factor_load * gamma, i)
                    qc.barrier()
                    for j in range(length):
                        p_comp2 = components[bus]["power"][j]
                        factor = 0.25 * p_comp1 * p_comp2
                        qc.rz(factor * gamma, i)
                        qc.rz(factor * gamma, j)
                        if i != j:
                            qc.rzz(factor * gamma, i, j)
                        qc.barrier()
                    #qc.rx(beta, i)
                    #for i in range(nqubits):
                    #    qc.rx(beta, i)

        # add mixing Hamiltonian to each qubit
        for i in range(nqubits):
            qc.rx(beta, i)

        qc.measure_all()

        return qc

    def create_qc3(self, components: dict, theta: list) -> QuantumCircuit:
        # 2 betas & 2 gammas
        """
        Creates a quantum circuit based on the components given and the cost function:

        Args:
            components: (dict) All components to be modeled as a Quantum Circuit.
            theta: (list) The optimizable values of the quantum circuit. Two arguments needed: beta = theta[0] and
                          gamma = theta[1].

        Returns:
            (QuantumCircuit) The created quantum circuit.
        """
        nqubits = len(components["qubit_map"])
        qc = QuantumCircuit(nqubits)

        beta = theta[0]
        gamma = theta[1]

        # add Hadamard gate to each qubit
        for i in range(nqubits):
            qc.h(i)
        qc.barrier()

        # add problem Hamiltonian for each bus
        index = 0
        for bus in components:
            if bus is not "qubit_map":
                beta = theta[2*index]
                gamma = theta[2*index+1]
                length = len(components[bus][f"flattened_{bus}"])
                for i in range(length):
                    p_comp1 = components[bus]["power"][i]
                    # negative load, since it removes power form the node
                    factor_load = -(components[bus]["load"]) * p_comp1
                    qc.rz(factor_load * gamma, i)
                    qc.barrier()
                    for j in range(length):
                        p_comp2 = components[bus]["power"][j]
                        factor = 0.25 * p_comp1 * p_comp2
                        qc.rz(factor * gamma, i)
                        qc.rz(factor * gamma, j)
                        if i != j:
                            qc.rzz(factor * gamma, i, j)
                        qc.barrier()
                    qc.rx(beta, i)
            index += 1

        # add mixing Hamiltonian to each qubit
        #for i in range(nqubits):
        #    qc.rx(beta, i)

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
            if bus is not "qubit_map":
                power -= components[bus]["load"]

                for comp in components[bus][f"flattened_{bus}"]:
                    i = components[bus][f"flattened_{bus}"].index(comp)
                    i_bit = components["qubit_map"][comp]
                    power += (components[bus]["power"][i] * float(bitstring[i_bit]))
                self.kirchhoff[f"rep{self.results_dict['iter_count']}"][bitstring][bus] = power
            power_total += abs(power)
        self.kirchhoff[f"rep{self.results_dict['iter_count']}"][bitstring]["total"] = power_total

        #if power_total is not 0:
        #    power_total += 5

        #return power_total ** 2
        return power_total

    def compute_expectation(self, counts: dict, components: dict) -> float:
        """
        Computes expectation value based on measurement results

        Args:
            counts: (dict) The bitstring is the key and its count the value.
            components: (dict) All components to be modeled as a Quantum Circuit.

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
        APIKEY = self.APItoken["IBMQ_API_token"]
        if simulate:
            if noise:
                # https://qiskit.org/documentation/apidoc/aer_noise.html
                IBMQ.save_account(APIKEY, overwrite=True)
                provider = IBMQ.load_account()
                # print(provider.backends())
                device = provider.get_backend("ibmq_lima")

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
            IBMQ.save_account(APIKEY, overwrite=True)
            provider = IBMQ.load_account()
            large_enough_devices = provider.backends(
                filters=lambda x: x.configuration().n_qubits > nqubits and not x.configuration().simulator)
            backend = least_busy(large_enough_devices)
            # backend = provider.get_backend("ibmq_lima")
            noise_model = None
            coupling_map = None
            basis_gates = None

        return backend, noise_model, coupling_map, basis_gates

    def get_expectation(self, components: dict, simulator: str = "aer_simulator", shots: int = 1024,
                        simulate: bool = True, noise: bool = False):
        """
        Builds the objective function, which can be used in a classical solver.

        Args:
            components: (dict) All components to be modeled as a Quantum Circuit
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
                                                                             nqubits=len(components["qubit_map"]))

        self.results_dict["shots"] = shots
        self.results_dict["simulate"] = simulate
        self.results_dict["noise"] = noise

        def execute_circ(theta):
            qc = self.create_qc2(components=components, theta=theta)
            self.results_dict["qc"] = qc.draw(output="latex_source")
            # qc.draw(output="latex")
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
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["backend"] = backend.configuration().to_dict()
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["theta"] = list(theta)
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["counts"] = counts
            self.kirchhoff[f"rep{self.results_dict['iter_count']}"] = {}

            return self.compute_expectation(counts=counts, components=components)

        return execute_circ


def main():
    testNetwork = pypsa.Network()
    # add node
    testNetwork.add("Bus", "bus1")
    testNetwork.add("Bus", "bus2")
    # add generators
    testNetwork.add("Generator", "Gen1", bus="bus1", p_nom=2, p_nom_extendable=False, marginal_cost=5)
    testNetwork.add("Generator", "Gen2", bus="bus2", p_nom=4, p_nom_extendable=False, marginal_cost=5)
    testNetwork.add("Generator", "Gen3", bus="bus2", p_nom=2, p_nom_extendable=False, marginal_cost=5)
    # line
    # p0= [-1,-2]
    # p1= [1, 2]
    # testNetwork.add("Line","line1",bus0="bus1", bus1="bus2",x=0.0001, s_nom=2, p0=p0, p1=p1)
    testNetwork.add("Line", "line1", bus0="bus1", bus1="bus2", x=0.0001, s_nom=2)
    testNetwork.add("Line", "line2", bus0="bus2", bus1="bus1", x=0.0001, s_nom=2)
    # add load
    testNetwork.add("Load", "load1", bus="bus1", p_set=2)
    testNetwork.add("Load", "load2", bus="bus2", p_set=2)

    #shots = 1024
    shots = 4096
    #shots = 16384
    simulator = "aer_simulator"  # UnitarySimulator, qasm_simulator, aer_simulator, statevector_simulator
    simulate = True
    noise = True
    #initial_guess = [1.0, 1.0]
    initial_guess = [1.0, 1.0, 1.0]
    #initial_guess = [1.0, 1.0, 1.0, 1.0]

    num_vars = len(initial_guess)

    loop_results = {}

    for i in range(1, 11):
        time_start = datetime.timestamp(datetime.now())
        print(i)

        qaoa = QaoaQiskit()
        spsa = SPSA(maxiter=200)

        components = qaoa.getComponents(network=testNetwork)

        expectation = qaoa.get_expectation(components=components,
                                           simulator=simulator,
                                           shots=shots,
                                           simulate=simulate,
                                           noise=noise)

        #res = spsa.minimize(fun=expectation, x0=initial_guess)
        #qaoa.results_dict["optimizeResults"]["x"] = list(res.x)  # solution [beta, gamma]
        #qaoa.results_dict["optimizeResults"]["fun"] = res.fun  # objective function value
        #qaoa.results_dict["optimizeResults"]["nfev"] = res.nfev  # number of objective function calls

        res = spsa.optimize(num_vars=num_vars, objective_function=expectation, initial_point=initial_guess)
        qaoa.results_dict["optimizeResults"]["x"] = list(res[0])  # solution [beta, gamma]
        qaoa.results_dict["optimizeResults"]["fun"] = res[1]  # objective function value
        qaoa.results_dict["optimizeResults"]["nfev"] = res[2]  # number of objective function calls

        qaoa.results_dict["initial_guess"] = initial_guess

        time_end = datetime.timestamp(datetime.now())
        duration = time_end - time_start
        qaoa.results_dict["duration"] = duration

        now = datetime.today()
        filename = f"Qaoa_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}_{now.microsecond}.json"
        with open(os.path.dirname(__file__) + "/../../results_qaoa/" + filename, "w") as write_file:
            json.dump(qaoa.results_dict, write_file, indent=2)
        filename2 = f"Kirchhoff_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}_{now.microsecond}.json"
        with open(os.path.dirname(__file__) + "/../../results_qaoa/" + filename2, "w") as write_file:
            json.dump(qaoa.kirchhoff, write_file, indent=2)

        last_rep = qaoa.results_dict["iter_count"]
        last_rep_counts = qaoa.results_dict[f"rep{last_rep}"]["counts"]
        loop_results[i] = {"filename": filename,
                           "optimize_Iterations": qaoa.results_dict["iter_count"],
                           "simulate": qaoa.results_dict["simulate"],
                           "noise": qaoa.results_dict["noise"],
                           "shots": shots,
                           "initial_guess": initial_guess,
                           "duration": duration,
                           "counts": last_rep_counts}

    now = datetime.today()
    filename = f"QaoaCompare_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}_{now.microsecond}.json"
    with open(os.path.dirname(__file__) + "/../../results_qaoa/qaoaCompare/" + filename, "w") as write_file:
        json.dump(loop_results, write_file, indent=2)
    # plot_histogram(qaoa.results_dict[f"rep{qaoa.results_dict['iter_count']}"]["counts"])
    # plt.show()


if __name__ == "__main__":
    main()
