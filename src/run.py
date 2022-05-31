"""This file is the entrypoint for the docker run command for the image build from the Dockerfile.
The bas image is `herrd1/siquan:latest`. The docker container loads the pypsa model and performs the
optimization of the unit commitment problem.

For this the docker run needs at least 2 and up to 3 arguments.
        sys.argv[1]: (str) path of the network file
        sys.argv[2]: (str) path of the config file
        sys.argv[3]: (str||None) string containing extra parameters that are not specified in config file
"""

import sys
from copy import deepcopy

from program import run


def main():
    # reading input
    network = sys.argv[1]
    params = sys.argv[2]
    # the extraParams string contains entries of a nested dictionary according to this rule
    # different config parameters are seperated using '--'
    # the keys of the different levels of one parameters are seperated using '-'
    # for each parameter, the last value is the value of the config, and the rest are keys of the nesting
    # any value found in extraParams will overwrite a value found in params
    extraParams = []
    extraParamsValues = []
    if len(sys.argv) >= 3 and sys.argv[3] != "":
        argumentSplit = [x.split("___") for x in sys.argv[3].split("____")]
        for [name, value] in argumentSplit:
            extraParams.append(name.split("__"))
            extraParamsValues.append(value)
        extraParamsValues = expandValueList(extraParamsValues)
        extraParamsValues = typecastValues(extraParamsValues)
    # run optimization
    if extraParamsValues:
        for values in extraParamsValues:
            response = run(data=network, params=params,
                           extraParams=extraParams,
                           extraParamValues=values)
            # save results
            response.save_to_json_local_docker()
    else:
        response = run(data=network, params=params)
        # save results
        response.save_to_json_local_docker()


def expandValueList(ValuesToExpand: list) -> list:
    """
    Takes a list of values for extra parameters, where each element in
    the list can represent multiple values for this parameter, and
    expands this list to return an list of list, where each possible
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
    for item in ValuesToExpand[1:]:
        itemList = item.split("__")
        if len(itemList) > 1:
            j = 0
            while j < len(valueLists):
                for i in range(len(itemList)-1):
                    valueLists.insert(valueLists.index(valueLists[j]),
                                      deepcopy(valueLists[j]))
                j += len(itemList)
        i = 0
        while i < len(itemList):
            j = i
            while j < len(valueLists):
                valueLists[j].append(itemList[i])
                j += len(itemList)
            i += 1
    return valueLists


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
    newValues = []
    for valueList in values:
        newValueList = []
        for value in valueList:
            if "[" in value:
                value = value.replace("[", "")
                value = value.replace("]", "")
                newValueList.append(list(value.split(" ")))
            else:
                try:
                    newValueList.append(int(value))
                except ValueError:
                    try:
                        newValueList.append(float(value))
                    except ValueError:
                        newValueList.append(value)
        newValues.append(newValueList)
    return newValues


if __name__ == "__main__":
    main()
