"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""
from typing import Dict, Any, Optional
from loguru import logger


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

# TODO elimate ${solver}Backend strings
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
}


def run(
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    extraParams: list = [],
) -> Response:

    response: Response
    try:
        # load all relevant data and parameters
        inputReader = InputReader(network=data, config=params, extraParams=extraParams)
        network = inputReader.getNetwork()

        # set up optimizer with input data
        OptimizerClass = ganBackends[inputReader.config["Backend"]]
        optimizer = OptimizerClass(reader=inputReader)
        # validate input in case there are restrictions like limited computation time
        # TODO: implement checks for different backends
        # optimizer.validateInput(path="Problemset", network=network)

        # run optimization
        transformedProblem = optimizer.transformProblemForOptimizer(network)
        solution = optimizer.optimize(transformedProblem)

        # hook for potential post processing like flow optimization for dwave solutions
        processedSolution = optimizer.processSolution(
            network, transformedProblem, solution
        )
        # return results
        # TODO: implement saving solution in Network and store in output dict.
        outputNetwork = optimizer.transformSolutionToNetwork(
            network, transformedProblem, processedSolution
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
