"""This file is the entrypoint for the docker run command for the image build from the Dockerfile.
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


def main():
    # reading input
    network = sys.argv[1]
    params = sys.argv[2]
    # the sys.argv[3] string contains entries of a nested dictionary,
    # where for each parameter its name and nested levels are separated by
    # "__", as well as all desired values for this parameter are separated
    # by "__". The names and values for parameters are separated by "___"
    # and between all name-value pairs a separator of "____" was inserted.
    extra_params = []
    extra_params_values = []
    # split up extra parameter string if parsed by Makefile
    if len(sys.argv) >= 3 and sys.argv[3] != "":
        argument_split = [x.split("___") for x in sys.argv[3].split("____")]
        for [name, value] in argument_split:
            extra_params.append(name.split("__"))
            extra_params_values.append(value)
        extra_params_values = expand_value_list(extra_params_values)
        extra_params_values = typecast_values(extra_params_values)
    # run optimization and save results
    for values in extra_params_values:
        response = run(data=network, params=params,
                       extra_params=extra_params,
                       extra_param_values=values)
        response.save_to_json_local_docker()
    if not extra_params_values:
        response = run(data=network, params=params)
        response.save_to_json_local_docker()


def expand_value_list(values_to_expand: list) -> list:
    """
    Takes a list of values for extra parameters, where each element in
    the list can represent multiple values for this parameter, and
    expands this list to return a list of list, where each possible
    combination of the input values is stored.
    E.g.:   Input:  ["1.0__2.0", "10__5__0"]
            Output: [["1.0", "10"], ["1.0", "5"], ["1.0", "0"],
                     ["2.0", "10"], ["2.0", "5"], ["2.0", "0"]]
    Args:
        values_to_expand: (list)
            A list of values for all extra parameters. The values for
            parameters have to be separated by "__".
            E.g.: ["1.0__2.0", "10__5__0"]

    Returns:
        (list)
            The expanded list of values. A list of lists with all
            possible combinations of the input values.
            E.g.: [["1.0", "10"], ["1.0", "5"], ["1.0", "0"],
                   ["2.0", "10"], ["2.0", "5"], ["2.0", "0"]]
    """
    value_lists = [[x] for x in values_to_expand[0].split("__")]
    return list(product(*value_lists))


def typecast_values(values: list) -> list:
    """
    Analyzes a of list of strings and typecasts them into floats,
    ints or list, if necessary.
    Args:
        values: (list)
            A list of lists of strings (returned by expand_value_list)
            which should be typecast.
            E.g.: [["1.0", "10", "test"], ["1.0", "5", "test"],
                   ["2.0", "10", "test"], ["2.0", "5", "test"]]
    Returns:
        (list)
            A list of lists with the typecast values.
            E.g.: [[1.0, 10, "test"], [1.0, 5, "test"],
                   [2.0, 10, "test"], [2.0, 5, "test"]]
    """
    return [[ast.literal_eval(literal)
                for literal in cross_product_element]
            for cross_product_element in values]


if __name__ == "__main__":
    main()
