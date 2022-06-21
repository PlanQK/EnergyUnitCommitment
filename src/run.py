"""This file is the entrypoint for the docker run command for the image build from the Dockerfile.
The base image is `herrd1/siquan:latest`. The docker container loads the pypsa model and performs the
optimization of the unit commitment problem. The result will be written to a json file  in a location
that the Makefile will mount to the host's drive.

For this the docker run needs at least 2 and up to 3 arguments.
        sys.argv[1]: (str) path of the network file
        sys.argv[2]: (str) path of the config file
        sys.argv[3]: (str||None) string containing extra parameters that are not specified in config file
"""

import sys
from copy import deepcopy

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
    extraParams = []
    extraParamsValues = []
    # split up extra parameter string if parsed by Makefile
    if len(sys.argv) >= 3 and sys.argv[3] != "":
        argumentSplit = [x.split("___") for x in sys.argv[3].split("____")]
        for [name, value] in argumentSplit:
            extraParams.append(name.split("__"))
            extraParamsValues.append(value)
        extraParamsValues = expandValueList(extraParamsValues)
        extraParamsValues = typecastValues(extraParamsValues)
    # run optimization and save results
    for values in extraParamsValues:
        response = run(data=network, params=params,
                       extraParams=extraParams,
                       extraParamValues=values)
        response.save_to_json_local_docker()
    if not extraParamsValues:
        response = run(data=network, params=params)
        response.save_to_json_local_docker()


def expandValueList(ValuesToExpand: list) -> list:
    """
    Takes a list of values for extra parameters, where each element in
    the list can represent multiple values for this parameter, and
    expands this list to return a list of list, where each possible
    combination of the input values is stored.
    E.g.:   Input:  ["1.0__2.0", "10__5__0"]
            Output: [["1.0", "10"], ["1.0", "5"], ["1.0", "0"],
                     ["2.0", "10"], ["2.0", "5"], ["2.0", "0"]]
    Args:
        ValuesToExpand: (list)
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
    valueLists = [[x] for x in ValuesToExpand[0].split("__")]
    return list(product(*valueLists))


def typecastValues(values: list) -> list:
    """
    Analyzes a list of list of strings and typecasts them into floats,
    ints or list, if necessary.
    Args:
        values: (list)
            A list of lists of strings (returned by expandValueList)
            which should be typecast.
            E.g.: [["1.0", "10", "test"], ["1.0", "5", "test"],
                   ["2.0", "10", "test"], ["2.0", "5", "test"]]
    Returns:
        (list)
            A list of lists with the typecast values.
            E.g.: [[1.0, 10, "test"], [1.0, 5, "test"],
                   [2.0, 10, "test"], [2.0, 5, "test"]]
    """
    return [ast.literal_eval(literal) for literal in values]


if __name__ == "__main__":
    main()
