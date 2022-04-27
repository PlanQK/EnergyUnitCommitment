"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""

import sys
from program import run


def main():

    inputData = sys.argv[1]
    network = sys.argv[2]
    if (len(sys.argv) <= 3 or sys.argv[3] == ""):
        extraParams = []
    else:
        extraParams = [ keyChain.split("-") for keyChain in sys.argv[3].split("--") ]

    response = run(data=network, params=inputData, extraParams=extraParams)

    response.save_to_json_local_docker()

    return


if __name__ == "__main__":
    main()
