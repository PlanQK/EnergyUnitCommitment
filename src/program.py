"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""
import libs.Backends as Backends
from libs.Backends.InputReader import InputReader
from libs.return_objects import Response, ResultResponse, ErrorResponse, ResultFileResponse
from typing import Dict, Any, Optional


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
    "qaoa": [Backends.QaoaQiskit, "QaoaBackend"]
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
