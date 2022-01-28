import networkx as nx
import matplotlib.pyplot as plt
import sympy
import json
import pypsa
import os.path
from datetime import datetime
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class QaoaQiskit():
    def __init__(self):
        self.results_dict = {"iter_count" : 0}


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
            counts: dict
                    key as bitstring, val as count

        Returns:
            avg: float
                  expectation value
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

    def get_expectation(self, components: dict, shots=512):
        """
        Runs parametrized circuit

        Args:
            G: networkx graph
            p: int,
                Number of repetitions of unitaries
        """

        backend = Aer.get_backend('aer_simulator')  # UnitarySimulator, qasm_simulator, aer_simulator
        backend.shots = shots


        def execute_circ(theta):
            qc = self.create_qc(components=components, theta=theta)
            self.results_dict["qc"] = qc.draw(output="latex_source")
            #qc.draw(output="latex")
            results = backend.run(qc, shots=shots).result()# seed_simulator=10
            counts = results.get_counts()
            self.results_dict["iter_count"] += 1
            self.results_dict[f"rep{self.results_dict['iter_count']}"] = {}
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["beta"] = theta[0]
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["gamma"] = theta[1]
            self.results_dict[f"rep{self.results_dict['iter_count']}"]["counts"] = counts

            return self.compute_expectation(counts=counts, components=components)

        return execute_circ


def main():
    qaoa = QaoaQiskit()

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

    components = qaoa.getBusComponents(network=testNetwork, bus="bus1")

    #components["power"] = [Parameter("x\u2081"), Parameter("x\u2082"), Parameter("x\u2083")]
    #qc_draw = qaoa.create_qc(components=components, theta=[Parameter("\u03B2"), Parameter("\u03B3")])
    #qc_draw.draw(output="mpl")
    #plt.show()

    loop_results = {}

    for i in range(1, 101):
        expectation = qaoa.get_expectation(components=components, shots=1024)

        res = minimize(fun=expectation, x0=[1.0, 1.0], method='COBYLA',
                       options={'rhobeg': 1.0, 'maxiter': 1000, 'tol': 0.0001, 'disp': False, 'catol': 0.0002})
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html#optimize-minimize-cobyla

        #store res as serializalbe in results_dict
        qaoa.results_dict["optimizeResults"] = dict(res)
        qaoa.results_dict["optimizeResults"]["x"] = res.x.tolist()
        qaoa.results_dict["optimizeResults"]["success"] = bool(res.success)

        now = datetime.today()
        filename = f"Qaoa_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}_{now.microsecond}.json"
        with open(os.path.dirname(__file__) + "/../../results_qaoa/" + filename, "w") as write_file:
            json.dump(qaoa.results_dict, write_file, indent=2)

        last_rep = qaoa.results_dict["optimizeResults"]["nfev"]
        last_rep_counts = qaoa.results_dict[f"rep{last_rep}"]["counts"]
        loop_results[i] = {"filename": filename,
                                "counts": last_rep_counts}

    now = datetime.today()
    filename = f"QaoaCompare_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}_{now.microsecond}.json"
    with open(os.path.dirname(__file__) + "/../../results_qaoa/qaoaCompare/" + filename, "w") as write_file:
        json.dump(loop_results, write_file, indent=2)
    #plot_histogram(qaoa.results_dict[f"rep{qaoa.results_dict['iter_count']}"]["counts"])
    #plt.show()


if __name__ == "__main__":
    main()