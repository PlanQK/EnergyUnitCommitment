"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""

import sys
import json, yaml
import random
import Backends
from Backends.InputReader import InputReader
from EnvironmentVariableManager import EnvironmentVariableManager


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
configs = {
    "classical": "SqaBackend",
    "sqa": "SqaBackend",
    "dwave-tabu": "DwaveBackend",
    "dwave-greedy": "DwaveBackend",
    "pypsa-glpk": "PypsaBackend",
    "pypsa-fico": "PypsaBackend",
    "dwave-hybrid": "DwaveBackend",
    "dwave-qpu": "DwaveBackend",
    "dwave-read-qpu": "DwaveBackend",
    "qaoa": "QaoaBackend",
    "test": "QaoaBackend"}

ganBackends = {
    "classical": Backends.ClassicalBackend,
    "sqa": Backends.SqaBackend,
    "dwave-tabu": Backends.DwaveTabuSampler,
    "dwave-greedy": Backends.DwaveSteepestDescent,
    "pypsa-glpk": Backends.PypsaGlpk,
    "pypsa-fico": Backends.PypsaFico,
    "dwave-hybrid": Backends.DwaveCloudHybrid,
    "dwave-qpu": Backends.DwaveCloudDirectQPU,
    "dwave-read-qpu": Backends.DwaveReadQPU,
    "qaoa": Backends.QaoaQiskit,
    "test": Backends.QaoaQiskit
}

def run(data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, extraParams: dict = None):

    input = InputReader(network=data, config=params)

    for key, value in extraParams.items():
        if key in input.config["IsingInterface"]:
            input.config["IsingInterface"][key] = value
        elif key in input.config[configs[input.config["Backend"]]]:
            input.config[configs[input.config["Backend"]]] = value
        else:
            raise ValueError(f"Extra parameter {key} not found in config.")

    OptimizerClass = ganBackends[input.config["Backend"]]

    optimizer = OptimizerClass(input)

    pypsaNetwork = input.getNetwork()

    # validate input has to throw and catch exceptions on it's own
    # TODO: possibly useless, as we get Network already in InputReader
    optimizer.validateInput(path="Problemset", network=pypsaNetwork)

    transformedProblem = optimizer.transformProblemForOptimizer(pypsaNetwork)

    solution = optimizer.optimize(transformedProblem)

    processedSolution = optimizer.processSolution(
        pypsaNetwork, transformedProblem, solution
    )
    outputNetwork = optimizer.transformSolutionToNetwork(
        pypsaNetwork, transformedProblem, processedSolution
    )

    if local docker flag:
        response = ResultFileResponse(result)
    else
        response = ResultResponse(result)
    or
        response = ErrorResponse

    response.send

    if envMgr["outputNetwork"]:
        outputNetwork.export_to_netcdf(
            f"Problemset/{str(envMgr['outputNetwork'])}"
        )

    with open(f"Problemset/{str(envMgr['outputInfo'])}", "w") as write_file:
        json.dump(optimizer.getOutput(), write_file, indent=2, default=str)
    return
