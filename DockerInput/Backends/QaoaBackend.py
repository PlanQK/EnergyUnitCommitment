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
from scipy.optimize import minimize

class QaoaQiskit():
    def __init__(self):
        self.results_dict = {"iter_count" : 0,
                             "simulate" : None,
                             "noise": None,
                             "shots" : None,
                             "components" : {},
                             "qc" : None,
                             "optimizeResults": {},
                             }

        with open(os.path.dirname(__file__) + "/../APItoken.json") as json_file:
            self.APItoken = json.load(json_file)

        self.backends = []


    def power_extraction(self, comp: str, components: dict, network: pypsa.Network) -> float:
        if comp in components["generators"]:
            return float(network.generators[network.generators.index == comp].p_nom)
        elif comp in components["positiveLines"]:
            return float(network.lines[network.lines.index == comp].s_nom)
        elif comp in components["negativeLines"]:
            return -float(network.lines[network.lines.index == comp].s_nom)


    def extract_power_list(self, components: dict, network: pypsa.Network) -> list:
        power_list = [0] * len(components["flattened"])
        for comp in components["flattened"]:
            i = components["flattened"].index(comp)
            power_list[i] = self.power_extraction(comp=comp, components=components, network=network)

        return power_list


    def getBusComponents(self, network: pypsa.Network, bus: str) -> dict:
        """return all labels of components that connect to a bus as a dictionary
        generators - at this bus
        loads - at this bus
        positiveLines - start in this bus
        negativeLines - end in this bus
        """
        components = {
            "generators":
                list(network.generators[network.generators.bus == bus].index),
            "positiveLines":
                list(network.lines[network.lines.bus1 == bus].index),
            "negativeLines":
                list(network.lines[network.lines.bus0 == bus].index),
            "load":
                sum(list(network.loads[network.loads.bus == bus].p_set)),
        }

        components["flattened"] = components["generators"] + components["positiveLines"] + components["negativeLines"]
        components["power"] = self.extract_power_list(components=components, network=network)

        self.results_dict["components"] = components

        return components


    def create_qc(self, components: dict, theta: list = [1, 2]) -> QuantumCircuit:
        nqubits = len(components["flattened"])
        qc = QuantumCircuit(nqubits)

        beta = theta[0]
        gamma = theta[1]

        # add Hadamard gate to each qubit
        for i in range(nqubits):
            qc.h(i)
        qc.barrier()

        # add problem Hamiltonian
        for i in range(nqubits):
            p_comp1 = components["power"][i]
            factor_load = -(components["load"]) * p_comp1  # negative load, since it removes power form the node
            qc.rz(factor_load * gamma, i)
            qc.barrier()
            for j in range(nqubits):
                p_comp2 = components["power"][j]
                factor = 0.25 * p_comp1 * p_comp2
                qc.rz(factor * gamma, i)
                qc.rz(factor * gamma, j)
                if i != j:
                    qc.rzz(factor * gamma, i, j)
                qc.barrier()

        # add mixing Hamiltonian
        for i in range(nqubits):
            qc.rx(beta, i)

        qc.measure_all()

        return qc


    def kirchhoff_satisfied2(self, bitstring: str, components: dict) -> float:
        power = - components["load"]

        for comp in components["flattened"]:
            i = components["flattened"].index(comp)
            power += (components["power"][i] * float(bitstring[i]))

        return abs(power)


    def kirchhoff_satisfied(self, bitstring: str, components: dict) -> float:
        power = 0
        for comp in components["flattened"]:
            i = components["flattened"].index(comp)
            if comp in components["generators"]:
                power += (components["power"][i] * float(bitstring[i]))
            elif comp in components["positiveLines"]:
                power += (components["power"][i] * float(bitstring[i]))
            elif comp in components["negativeLines"]:
                power -= (components["power"][i] * float(bitstring[i]))
        if power != components["load"]:
            return 1  # correct solutions are 1 --> counts are added to the overall result, therefore acting negatively on the minimization of the expectation value
        else:
            return 0  # correct solutions are 0 --> counts are not added to the overall result, therefore acting positively on the minimization of the expectation value

    def compute_expectation(self, counts: dict, components: dict):
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
            self.results_dict[f"rep{self.results_dict['iter_count']}"][f"{bitstring}"] = {"count": count,
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
            (list??) The coupling map of the chosen backend, if noise is set to True.
            (NoiseModel.basis_gates) The basis gates of the noise model, if noise is set to True.
        """
        APIKEY = self.APItoken["IBMQ_API_token"]
        if simulate:
            if noise:
                # https://qiskit.org/documentation/apidoc/aer_noise.html
                IBMQ.save_account(APIKEY, overwrite=True)
                provider = IBMQ.load_account()
                #print(provider.backends())
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
            #backend = provider.get_backend("ibmq_lima")
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
                                                                             nqubits=len(components["flattened"]))

        self.results_dict["shots"] = shots
        self.results_dict["simulate"] = simulate
        self.results_dict["noise"] = noise

        def execute_circ(theta):
            qc = self.create_qc(components=components, theta=theta)
            self.results_dict["qc"] = qc.draw(output="latex_source")
            #qc.draw(output="latex")
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
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["beta"] = float(theta[0])
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["gamma"] = float(theta[1])
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["counts"] = counts

            self.backends.append(backend.name())

            return self.compute_expectation(counts=counts, components=components)

        return execute_circ


def main():
    testNetwork = pypsa.Network()
    # add node
    testNetwork.add("Bus", "bus1")
    testNetwork.add("Bus", "bus2")
    # add generators
    testNetwork.add("Generator", "Gen1", bus="bus1", p_nom=1, p_nom_extendable=False, marginal_cost=5)
    testNetwork.add("Generator", "Gen2", bus="bus2", p_nom=2, p_nom_extendable=False, marginal_cost=5)
    # line
    # p0= [-1,-2]
    # p1= [1, 2]
    # testNetwork.add("Line","line1",bus0="bus1", bus1="bus2",x=0.0001, s_nom=2, p0=p0, p1=p1)
    testNetwork.add("Line", "line1", bus0="bus1", bus1="bus2", x=0.0001, s_nom=2)
    testNetwork.add("Line", "line2", bus0="bus2", bus1="bus1", x=0.0001, s_nom=2)
    # add load
    testNetwork.add("Load", "load1", bus="bus1", p_set=2)
    testNetwork.add("Load", "load2", bus="bus2", p_set=1)



    #components["power"] = [Parameter("x\u2081"), Parameter("x\u2082"), Parameter("x\u2083")]
    #qc_draw = qaoa.create_qc(components=components, theta=[Parameter("\u03B2"), Parameter("\u03B3")])
    #qc_draw.draw(output="mpl")
    #plt.show()

    shots = 1024
    simulator = "aer_simulator"  # UnitarySimulator, qasm_simulator, aer_simulator, statevector_simulator
    simulate = True
    noise = True
    initial_guess = [1.0, 1.0]

    loop_results = {}

    for i in range(1, 101):
        print(i)

        qaoa = QaoaQiskit()
        spsa = SPSA(maxiter=100)

        components = qaoa.getBusComponents(network=testNetwork, bus="bus1")

        expectation = qaoa.get_expectation(components=components,
                                           simulator=simulator,
                                           shots=shots,
                                           simulate=simulate,
                                           noise=noise)
        res = spsa.optimize(num_vars=2, objective_function=expectation, initial_point=initial_guess)
        #res = minimize(fun=expectation, x0=initial_guess, method='COBYLA',
        #               options={'rhobeg': 1.0, 'maxiter': 1000, 'tol': 0.0001, 'disp': False, 'catol': 0.0002})
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html#optimize-minimize-cobyla

        #store res as serializalbe in results_dict
        #qaoa.results_dict["optimizeResults"] = dict(res) # if use COBYLA
        #qaoa.results_dict["optimizeResults"]["x"] = res.x.tolist() # if use COBYLA
        #qaoa.results_dict["optimizeResults"]["success"] = bool(res.success) # if use COBYLA
        qaoa.results_dict["optimizeResults"]["x"] = list(res[0])
        qaoa.results_dict["optimizeResults"]["fun"] = res[1]
        qaoa.results_dict["optimizeResults"]["nfev"] = res[2]

        now = datetime.today()
        filename = f"Qaoa_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}_{now.microsecond}.json"
        with open(os.path.dirname(__file__) + "/../../results_qaoa/" + filename, "w") as write_file:
            json.dump(qaoa.results_dict, write_file, indent=2)

        last_rep = qaoa.results_dict["iter_count"]
        last_rep_counts = qaoa.results_dict[f"rep{last_rep}"]["counts"]
        loop_results[i] = {"filename" : filename,
                           "optimize_Iterations": qaoa.results_dict["iter_count"],
                           "backends": qaoa.backends,
                           "simulate" : qaoa.results_dict["simulate"],
                           "noise": qaoa.results_dict["noise"],
                           "shots" : shots,
                           "counts" : last_rep_counts}

    now = datetime.today()
    filename = f"QaoaCompare_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}_{now.microsecond}.json"
    with open(os.path.dirname(__file__) + "/../../results_qaoa/qaoaCompare/" + filename, "w") as write_file:
        json.dump(loop_results, write_file, indent=2)
    #plot_histogram(qaoa.results_dict[f"rep{qaoa.results_dict['iter_count']}"]["counts"])
    #plt.show()


if __name__ == "__main__":
    main()