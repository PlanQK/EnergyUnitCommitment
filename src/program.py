"""This file provides the entrypoint for starting an optimization run via the 
function `run`. It returns a response object containing information on the result
of the optimization. The PlanQK service uses this method to generate the response
to a request and the local execution of an optimization uses the provided values
in the makefile to call this method."""

from typing import Union

import pypsa
from loguru import logger

from os import environ

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

# TODO eliminate ${solver}Backend strings
gan_backends = {
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
        data: Union[pypsa.Network, str] = None,
        params: Union[dict, str] = None,
        extra_params: list = [],
        extra_param_values: list = [],
) -> Response:
    """
    This is the entrypoint for starting an optimization run. data is used to provide
    the network and params the default way to give solver parameterns. These are used
    by the PlanQK service. The other parameters are hooks for the Makefile to change
    config entries of the config file on the fly.

    Args:
        data: (pypsa.Network|str) the network to be optimized. This is either
                a pypsa network, a dict containing a serialized pypsa network
                or the path to a netcdf file containing the network
        params: (dict|str) The default argument for giving optimizer parameters.
                They are either given directly as a dict, or a path to a config file
        extra_params: (list) a list containing parameter names of params to be overwritten
                This is exclusively used by the Makefile.
        extra_param_values: (list) a list containing the values of the parameters to
                be overwritten as specified by extra_params
    Returns:
        (ResultResponse) The response that contains the meta data
                         and results of the optimization
    """
    response: Response
    try:
        # load all relevant data and parameters
        input_reader = InputReader(network=data, config=params,
                                   extra_params=extra_params,
                                   extra_param_values=extra_param_values)

        # set up optimizer with input data
        optimizer_class = gan_backends[input_reader.config["backend"]]
        optimizer = optimizer_class(reader=input_reader)

        # run optimization
        optimizer.transform_problem_for_optimizer()
        # disable runtime limits on the optimization by setting the env variable
        # in the dockerfile. This limits runtime on the platform, but allows you
        # to run as long as you want locally
        if not environ.get("TRUSTED_USER", False):
            optimizer.check_input_size()
        optimizer.optimize()

        # hook for potential post-processing like flow optimization for dwave
        # solutions
        optimizer.process_solution()

        optimizer.transform_solution_to_network()

        output = optimizer.get_output()
        logger.info("Calculation successfully executed")
        return ResultResponse(result=output)
    except Exception as e:
        error_code = "500"
        error_detail = f"{type(e).__name__}: {e}"
        logger.info("An error occurred")
        logger.info(f"error code: {error_code}")
        logger.info(f"details: {error_detail}")
        return ErrorResponse(code=error_code, detail=error_detail)
