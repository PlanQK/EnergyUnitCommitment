#!/bin/bash

# This script can be used for testing the planqk service. 

# If you want to make the service on the actual platform
# you have to slightly adjust the PlanQK_requirements.txt file, changing the name to `requirements.txt`
# Currently, the requirements doesn't use pip to install pypsa because it needs a patch in order to read
# networks that are stored as json files.

# How to use
# Use the option -b to rebuild the docker image before running it. It takes no further arguments.
# You have to change the NETWORK and CONFIG variable to point to the serialized network and config
# respectively

# this doesn't work with qaoa because when dumping the json, it sorts. qaoa's result dict has both
# integer and string keys. Only the repetition number is an int.


DOCKERTAG=energy:planqk
GIT_ROOT=`git rev-parse --show-toplevel`

NETWORK=220124cost5input_10_0_20
NETWORK=testNetwork4QubitIsing_2_0_20
CONFIG=config

if [[ $1 = -b ]]
then
    docker build -f ${GIT_ROOT}/PlanQK_Dockerfile -t ${DOCKERTAG} ${GIT_ROOT}
fi


docker run -it \
-e BASE64_ENCODED=false \
-e LOG_LEVEL=DEBUG \
-v ${GIT_ROOT}/input/${NETWORK}.json:/var/input/data/data.json \
-v ${GIT_ROOT}/input/${CONFIG}.json:/var/input/params/params.json \
${DOCKERTAG}  | tee  >(tail -n 1 > JobResponse)

exit 0
