f"""This file is the entrypoint for the docker run command for the image build from the Dockerfile.
The base image is `herrd1/siquan:latest`. The docker container loads the pypsa model and performs the
optimization of the unit commitment problem. The result will be written to a json file  in a location
that the Makefile will mount to the host's drive. In order to do that, it transforms the arguments
to call the `run` method, which will return an object containing the information of the optimization.

For this the docker run needs at least 2 and up to 3 arguments.
        sys.argv[1]: (str) path of the network file
        sys.argv[2]: (str) path of the config file
        sys.argv[3]: (str||None) string containing extra parameters that are not specified in config file
"""

import sys

from program import run

from itertools import product
import ast

keyword_seperator = "__"
params_seperator = "____"

def main():
    # reading input
    network = sys.argv[1]
    params = sys.argv[2]
    # the sys.argv[3] string contains configuration parameter so the makefile
    # can also pass parameters
    cli_params_dict = {}
    if len(sys.argv) > 3 :
        cli_params_dict = parse_cli_params(sys.argv[3])

    response = run(data=network, params=params, params_dict=cli_params_dict)
    response.dump_results()
    
def parse_cli_params(param_string: str):
    """
    Parse the input of the command line that contains configuration values

    Takes a string containing a parameter and which values you want to use
    and parses that into a pair of lists. The first list containts the list
    of keys which to descent into the config dictionary, and the second list
    containts the list of python values (str, int, float) that are going to be used

    Args:
        param_string: (str)
            A string containing entries of a nested dictionary to be parsed
            Different Parameters are seperated by `params_seperator`. The keys
            and the value are seperated by `keyword_seperator`
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

def insert_value(key_chain, value, current_level):
    """
    insert a value in the dictionary by descending the keychain

    Args:
        key_chain: (list)
            A list of strings that are the keys in a nested dictionary
        value: (any)
            The value to be written into the dictionary
        dictionary: (dictO
            The dictionary in which to write the value

    Returns:
        Modifies the passed dictionary
    """
    for key in key_chain[:-1]:
        current_level = current_level.setdefault(key, {})
    current_level[key_chain[-1]] = value


if __name__ == "__main__":
    main()
