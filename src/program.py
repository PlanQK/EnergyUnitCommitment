"""This file provides the entrypoint for starting an optimization run via the
function `run`. It returns a response object containing information on the result
of the optimization. The PlanQK service uses this method to generate the response
to a request and the local execution of an optimization uses the provided values
in the makefile to call this method."""

from typing import Union
import traceback
from os import environ

import pypsa
from loguru import logger


try:
    # import when building image for local use
    import libs.backends
    from libs.backends.input_reader import InputReader
    from libs.return_objects import Response, ResultResponse, ErrorResponse
except ImportError:
    # fall back to relative import. Necessary for the PlanQK service ##
    from .libs import backends
    from .libs.backends.input_reader import InputReader
    from .libs.return_objects import Response, ResultResponse, ErrorResponse


def run(
    data: Union[pypsa.Network, dict, str] = None,
    params: Union[dict, str] = None,
    params_dict: dict = None,
) -> Response:
    """
    This is the entrypoint for starting an optimization run. data is used to provide
    the network and params the default way to give solver parameters. These are used
    by the PlanQK service. The other parameter is a hook for the Makefile to change
    config entries of the config file on the fly.

    Args:
        data: (pypsa.Network|str)
            The network to be optimized. This is either a pypsa network, a dictionary
            containing a serialized pypsa network or the path to a netcdf file
            containing the network
        params: (dict|str)
            The default argument for giving optimizer parameters. They are either
            given directly as a dict, or a path to a config file
        params_dict: (dict)
            A python dictionary containing configuration parameters that are
            added after the params argument has been processed. This allows overwriting
            values that were passed in the config file that was read in params
    Returns:
        (ResultResponse)
            The response that contains the metadata and results of the optimization
    """
    try:
        # load all relevant data and parameters
        input_reader = InputReader(network=data, config=params, params_dict=params_dict)

        # set up optimizer with input data and the passed configuration
        optimizer_class = input_reader.get_optimizer_class()
        optimizer = optimizer_class.create_optimizer(reader=input_reader)

        # run the optimization

        # perform preprocessing steps like constructing the ising model
        optimizer.transform_problem_for_optimizer()
        # disable runtime limits on the optimization by setting the env variable
        # in the dockerfile. This limits runtime on the platform, but allows you
        # to run as long as you want locally
        if environ.get("TRUSTED_USER", False) != "Yes":
            print("Runtime is limited, checking problem size")
            optimizer.check_input_size(limit=15)
        optimizer.optimize()

        # hook for potential post-processing like flow optimization for d-wave
        # solutions
        optimizer.process_solution()

        # get the results from the optimizer and display key information
        optimizer.print_report()
        output = optimizer.get_output()
        logger.info("Calculation successfully executed")
        return ResultResponse(result=output)
    except Exception as e:
        logger.error(
            "An error occurred while processing. Error reads:"
            "\n" + traceback.format_exc()
        )
        return ErrorResponse(code="500", detail=f"{type(e).__name__}: {e}")
