"""This file is the entrypoint for the docker run command for the image build from the Dockerfile.
The bas image is `herrd1/siquan:latest`. The docker container loads the pypsa model and performs the
optimization of the unit commitment problem.

For this the docker run needs at least 2 and up to 3 arguments.
        sys.argv[1]: (str) path of the network file
        sys.argv[2]: (str) path of the config file
        sys.argv[3]: (str||None) string containing extra parameters that are not specified in config file
"""

import sys
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
    if len(sys.argv) <= 3 or sys.argv[3] == "":
        extraParams = []
    else:
        extraParams = [keyChain.split("-") for keyChain in sys.argv[3].split("--")]
    # run optimization
    response = run(data=network, params=params, extraParams=extraParams)
    # save reusults
    response.save_to_json_local_docker()


if __name__ == "__main__":
    main()
