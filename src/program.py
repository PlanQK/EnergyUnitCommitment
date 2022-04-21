"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""
import libs.Backends as Backends
from libs.Backends.InputReader import InputReader
from libs.return_objects import Response, ResultResponse, ErrorResponse, ResultFileResponse
from typing import Dict, Any, Optional

FOLDER = "Problemset"


errorMsg = """
Usage: run.py [classical | sqa | dwave-tabu | dwave-greedy | dwave-hybrid | dwave-qpu]
Arguments:
    classical: run a classical annealing algorithm
    sqa: run the discrete time simulated quantum annealing algorithm
    dwave-tabu: run the classical optimization procedure locally using the dwave package (tabu-search)
    dwave-greedy: run the classical optimization procedure locally using the dwave package (greedy)
    dwave-hybrid: run a hybrid optimization procedure from dwave (through cloud)
    dwave-qpu: run the optimization on a dwave quantum annealing device (through cloud)
    dwave-read-qpu: reuse the optimization of a dwave quantum annealing device (read from local drive)


Any further settings are specified through environment variables:
    optimizationCycles: 1000  Number of optimization cycles
            (each spin gets an update proposal once during one cycle)
    temperatureSchedule: []  (for sqa) how the temperature changes during the annealing run
    transverseFieldSchedule: [] (for sqa) how the transverse field changes during the annealing run
    cubicConstraints: false  DWave does not support cubic constraints

"""
ganBackends = {
    "classical": [Backends.ClassicalBackend, "SqaBackend"],
    "sqa": [Backends.SqaBackend, "SqaBackend"],
    "dwave-tabu": [Backends.DwaveTabuSampler, "DwaveBackend"],
    "dwave-greedy": [Backends.DwaveSteepestDescent, "DwaveBackend"],
    "pypsa-glpk": [Backends.PypsaGlpk, "PypsaBackend"],
    "pypsa-fico": [Backends.PypsaFico, "PypsaBackend"],
    "dwave-hybrid": [Backends.DwaveCloudHybrid, "DwaveBackend"],
    "dwave-qpu": [Backends.DwaveCloudDirectQPU, "DwaveBackend"],
    "dwave-read-qpu": [Backends.DwaveReadQPU, "DwaveBackend"],
    "qaoa": [Backends.QaoaQiskit, "QaoaBackend"],
    "test": [Backends.QaoaQiskit, "QaoaBackend"]
}


def run(data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None,
        storeFile: bool = False, extraParams: dict = None):

    response: Response
    try:

        inputReader = InputReader(network=data, config=params)

        # TODO: key naming in IsingInterface config has to be unique!!!
        inputReader = add_extra_parameters(reader=inputReader, params=extraParams,
                                           backend=ganBackends[inputReader.config["Backend"]][1])

        OptimizerClass = ganBackends[inputReader.config["Backend"]][0]

        optimizer = OptimizerClass(reader=inputReader)

        pypsaNetwork = inputReader.getNetwork()

        # validate input has to throw and catch exceptions on it's own
        # TODO: possibly useless, as we get Network already in InputReader and check if it is a Pypsa.network
        optimizer.validateInput(path="Problemset", network=pypsaNetwork)

        transformedProblem = optimizer.transformProblemForOptimizer(pypsaNetwork)

        solution = optimizer.optimize(transformedProblem)

        # TODO: possibly useless? Process what?
        processedSolution = optimizer.processSolution(
            pypsaNetwork, transformedProblem, solution
        )

        # TODO: implement saving solution in Network and store in output dict.
        outputNetwork = optimizer.transformSolutionToNetwork(
            pypsaNetwork, transformedProblem, processedSolution
        )

        output = optimizer.getOutput()

        if storeFile:
            response = ResultFileResponse(result=output)
        else:
            response = ResultResponse(result=output)
    except Exception as e:
        error_message = e
        response = ErrorResponse(500, f"{type(e).__name__}: {error_message}")

    response.send()


def add_extra_parameters(reader: InputReader, params: dict, backend: str) -> InputReader:
    if params is not None:
        for key, value in params.items():
            if is_number(value):
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            if key in reader.config[backend]:
                reader.config[backend] = value
            elif key in reader.config["IsingInterface"]:
                reader.config["IsingInterface"][key] = value
            elif key in reader.config["IsingInterface"]["kirchhoff"]:
                reader.config["IsingInterface"]["kirchhoff"][key] = value
            elif key in reader.config["IsingInterface"]["marginalCost"]:
                reader.config["IsingInterface"]["marginalCost"][key] = value
            elif key in reader.config["IsingInterface"]["minUpDownTime"]:
                reader.config["IsingInterface"]["minUpDownTime"][key] = value
            else:
                raise ValueError(f"Extra parameter {key} not found in config.")

    return reader


def is_number(n: str) -> bool:
    try:
        float(n)
    except ValueError:
        return False
    return True
