#!/bin/bash

# This script can be used for testing the planqk service. It uses a deprecated way to start the container
# by giving the parameters as arguments of the docker run command. This limits the input size by because
# the shell can't take arbitrarily large arguments. Use `testWithVolumeInput.sh` to test with bigger 
# problem instances. 

# If you want to make the service on the actual platform you have to slightly adjust the 
# PlanQK_requirements.txt file, changing the name to `requirements.txt`. Currently, the requirements 
# doesn't use pip to install pypsa because it needs a patch in order to read networks that are stored 
# as json files.

# How to use
# Use the option -b to rebuild the docker image before running it. Then you have to give the path of the json
# file that contains both the serialized network and the params to be used. This can be done by either
# changing the values of the variables below, or give it as the last argument relative to
# `{GIT_ROOT}/${PATH_PREFIX}`

# this doesn't work with qaoa because when dumping the json, it sorts. qaoa's result dict has both
# integer and string keys. Only the repetition number is an int.


GIT_ROOT=`git rev-parse --show-toplevel`
PATH_PREFIX="/input/"
PATH_TO_JSON="testService"
DOCKERTAG=energy:planqk

file="${GIT_ROOT}/${PATH_PREFIX}${PATH_TO_JSON}.json"

while [ ! -z "$1" ];
do
    case $1 in

        -b)
            docker build -f ${GIT_ROOT}/PlanQK_Dockerfile -t ${DOCKERTAG} ${GIT_ROOT}
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

echo "jq done"

unameOut=$(uname -s)

case $unameOut in
    Linux)
        docker run -it -e INPUT_DATA="${data}" -e INPUT_PARAMS="${params}" ${DOCKERTAG}
        ;;
    *)
        winpty docker run -it -e INPUT_DATA="${data}" -e INPUT_PARAMS="${params}" ${DOCKERTAG}
        ;;
esac


