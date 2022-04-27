"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""
from typing import Dict, Any, Optional

from loguru import logger

## import when debugging locally ##

try:
    # import when building image for local use
    import libs.Backends as Backends
    from libs.Backends.InputReader import InputReader
    from libs.return_objects import Response, ResultResponse, ErrorResponse
except ImportError:
    # fall back to relative import when using PlanQK docker ##
    from .libs import Backends
    from .libs.Backends.InputReader import InputReader
    from .libs.return_objects import Response, ResultResponse, ErrorResponse



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
        extraParams: list = []) -> Response:

    response: Response
    try:

        inputReader = InputReader(network=data, config=params)

        inputReader.addExtraParameters(extraParams)

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
        logger.info("Calculation successfully executed")
        return ResultResponse(result=output)
    except Exception as e:
        error_code = "500"
        error_detail = f"{type(e).__name__}: {e}"
        logger.info("An error occurred")
        logger.info(f"error code: {error_code}")
        logger.info(f"details: {error_detail}")
        return ErrorResponse(code=error_code, detail=error_detail)


