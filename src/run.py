"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""

import sys
from program import run


def main():

    if len(sys.argv) == 2:
        inputData = sys.argv[1]
        network = sys.argv[2]
        param = None
    else:
        inputData = sys.argv[1]
        network = sys.argv[2]
        param = sys.argv[3]

    extraParams = {}
    if param:
        paramList = param.split("_")
        for item in paramList:
            splitItem = item.split("-")
            extraParams[splitItem[0]] = splitItem[1]

    run(data=network, params=inputData, storeFile=True, extraParams=extraParams)

    return


if __name__ == "__main__":
    main()
