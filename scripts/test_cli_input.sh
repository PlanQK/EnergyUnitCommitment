#!/bin/bash

# This script can be used for testing the planqk service. It uses a deprecated way to start the container
# by giving the parameters as arguments of the docker run command. This limits the input size by because
# the shell can't take arbitrarily large arguments. Use `test_volume_input.sh` to test with bigger 
# problem instances. 

# Build PlanQK service
# --------------------
# Zip the folder `src/` with the `requirements.txt` and upload it to the platform. If you want to build the 
# image locally for testing, run `scripts/test_volume_input.sh` with the option `-b`

# How to use
# -----------
# Use the option -b to rebuild the docker image before running it. Then you have to give the path of the json
# file that contains both the serialized network and the params to be used. This can be done by either
# changing the values of the variables below, or give it as the last argument relative to
# `{GIT_ROOT}/${PATH_PREFIX}`

# set up some variables
GIT_ROOT=`git rev-parse --show-toplevel`
DOCKERTAG=energy:planqk
SERVICE_DOCKERFILE=PlanQK_Dockerfile

# default path where json files are stored
PATH_PREFIX="input/"

# You can change this file name or pass it as the last argument when calling the script
PATH_TO_JSON_FILE="test_service_input"

file="${GIT_ROOT}/${PATH_PREFIX}${PATH_TO_JSON_FILE}.json"

while [ ! -z "$1" ];
do
    case $1 in

        -b)
            docker build -f ${GIT_ROOT}/${PATH_PREFIX}${SERVICE_DOCKERFILE} -t ${DOCKERTAG} ${GIT_ROOT}
            ;;

        *)
            file="${GIT_ROOT}/${PATH_PREFIX}${1}.json"
            ;;
    esac
    shift
done

echo "using ${file} as input"

data=`cat $file | jq '.data' | base64`
params=`cat $file | jq '.params' | base64`

echo "parsing json using jq done"

# use winpty if you run it on windows
uname_out=$(uname -s)

case $uname_out in
    Linux)
        docker run -it -e INPUT_DATA="${data}" -e INPUT_PARAMS="${params}" -e LOG_LEVEL=DEBUG ${DOCKERTAG}
        ;;
    *)
        winpty docker run -it -e INPUT_DATA="${data}" -e INPUT_PARAMS="${params}" -e LOG_LEVEL=DEBUG ${DOCKERTAG}
        ;;
esac

exit
