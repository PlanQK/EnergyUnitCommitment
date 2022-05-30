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
    if len(sys.argv) <= 4 or sys.argv[3] == "" or sys.argv[4] == "":
        extraParams = []
        extraParamsValues = []
    else:
        print(f"Extra Parameters raw: {sys.argv[3]}")
        extraParams = [x.split("_") for x in sys.argv[3].split("__")]
        print(f"Extra Parameter Values raw: {sys.argv[4]}")
        extraParamsValues = expandValueList(sys.argv[4].split("__"))
        print(f"Extra Parameter Values preTypeCast: {extraParamsValues}")
        extraParamsValues = typecastValues(values=extraParamsValues)
    # run optimization
    print(f"Extra Parameter: {extraParams}")
    print(f"Extra Parameter Values: {extraParamsValues}")
    response = run(data=network, params=params,
                   extraParams=extraParams, extraParamValues=extraParamsValues)
    # save results
    response.save_to_json_local_docker()


def expandValueList(ValuesToExpand: list) -> list:
    valueLists = [[x] for x in ValuesToExpand[0].split("_")]
    for item in ValuesToExpand[1:]:
        itemList = item.split("_")
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
    newValues = []
    for valueList in values:
        newValueList = []
        for value in valueList:
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
