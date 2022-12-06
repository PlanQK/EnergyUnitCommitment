#!/bin/bash

# This script is for testing the planqk service using json files in the input folder

# Build PlanQK service
# ------------------
# Zip the folder `src/` with the `requirements. If you want to build the image locally,
# run this script with the option `-b`

# How to use
# ----------
# Use the option -b to rebuild the docker image before running it. It takes no 
# further arguments. You have to change the NETWORK and CONFIG variable to point 
# to the serialized network and config respectively

# set up some variables
GIT_ROOT=`git rev-parse --show-toplevel`
DOCKERTAG=energy:planqk
SERVICE_DOCKERFILE=input/PlanQK_Dockerfile

# set up input network and parameter. 
NETWORK=defaultnetwork
NETWORK=toy_network

CONFIG=config

# build the service image if the option -b was given
if [[ $1 = -b ]]
then
    docker build -f ${GIT_ROOT}/${SERVICE_DOCKERFILE} -t ${DOCKERTAG} ${GIT_ROOT}
fi

# run service test
docker run -it \
    -e BASE64_ENCODED=false \
    -e LOG_LEVEL=DEBUG \
    -e RUNNING_IN_DOCKER=Yes \
    -v ${GIT_ROOT}/input/${NETWORK}.json:/var/input/data/data.json \
    -v ${GIT_ROOT}/input/${CONFIG}.json:/var/input/params/params.json \
    ${DOCKERTAG}  | tee  >(tail -n 1 > results_general_sweep/job_response)

exit 0
