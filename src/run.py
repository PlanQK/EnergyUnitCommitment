"""This file is the entrypoint for the docker run command of the docker image build from the Dockerfile.
The base image is `herrd1/siquan:latest`, which contains the simulated quantum annealing solver. 
The docker container loads the pypsa model and performs the optimization of the unit commitment problem. 
The result will be written to a json file  in a location that the Makefile will mount to the host's drive.
In order to do that, it transforms the arguments to call the `run` method, which will return an object 
containing the information of the optimization.

For this the docker run needs at least 2 and up to 3 arguments.
        sys.argv[1]: (str) path of the network file
        sys.argv[2]: (str) path of the config file
        sys.argv[3]: (str|None) A string containing extra parameters that are not specified in config file
"""

import sys
import ast

from program import run


def main():
    # reading input files
    network = sys.argv[1]
    params = sys.argv[2]
    # the sys.argv[3] string contains configuration parameter so the makefile
    # can also pass parameters
    cli_params_dict = {}
    if len(sys.argv) > 3:
        cli_params_dict = parse_cli_params(sys.argv[3])

    response = run(data=network, params=params, params_dict=cli_params_dict)
    # dump the result json in `problemset/`, which has to mounted to the host
    response.dump_results(folder="problemset/")


def parse_cli_params(param_string: str,
                     keyword_seperator: str = "__",
                     params_seperator: str = "____",) -> dict:
    """
    Parse the input of the command line that contains configuration values

    Takes the string and converts into a nested dict. Different entries are
    separated by the default separator `"____"` and the keys for the different
    levels are separated by the default separator `"__"`.

    Args:
        param_string: (str)
            A string containing entries of a nested dictionary to be parsed.
        keyword_seperator: (str)
            The string by which to split the string containing the nested keys
        params_seperator: (str)
            The string by which to split the various parameters, that will be
            inserted into the nested dictionary

    Returns: (dict)
        A nested dictionary of configuration entries
    """
    if not param_string:
        return {}
    result = {}
    param_string_list = param_string.split(params_seperator)
    for param_string in param_string_list:
        entries = param_string.split(keyword_seperator)
        key_chain, value = entries[:-1], entries[-1]
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass
        insert_value(key_chain, value, result)
    return result


def insert_value(key_chain: list, 
                 value: any,
                 current_level: dict) -> None:
    """
    Insert the passed value into the dictionary by descending 
    the list of keys

    Args:
        key_chain: (list)
            A list of strings that are the keys in a nested dictionary
        value: (any)
            The value to be written into the dictionary
        current_level: (dict)
            The dictionary in which to write the value

    Returns: (None)
        Modifies the passed dictionary and returns `None`.
    """
    for key in key_chain[:-1]:
        current_level = current_level.setdefault(key, {})
    current_level[key_chain[-1]] = value


if __name__ == "__main__":
    main()
