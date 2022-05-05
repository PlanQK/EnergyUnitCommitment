import json
import yaml
import pypsa
import os.path

from libs.Backends.InputReader import InputReader
from libs.Backends.QaoaBackend import QaoaQiskit


def main():
    inputNet = "testNetwork4QubitIsing_2_0_20.nc"
    configFile = "config.yaml"

    netImport = pypsa.Network(
        os.path.dirname(__file__) + "../../sweepNetworks/" + inputNet
    )

    with open(os.path.dirname(__file__) + "/Configs/" + configFile) as file:
        config = yaml.safe_load(file)

    inputReader = InputReader(network=netImport, config=config)

    qaoa = QaoaQiskit(reader=inputReader)
    components = qaoa.transformProblemForOptimizer()

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

    qaoa.optimize(transformedProblem=components)

    output = qaoa.getOutput()

    filename = qaoa.output["file_name"]
    with open(
        os.path.dirname(__file__) + "../../results_local_test/" + filename, "w"
    ) as write_file:
        json.dump(output, write_file, indent=2, default=str)


if __name__ == "__main__":
    main()
