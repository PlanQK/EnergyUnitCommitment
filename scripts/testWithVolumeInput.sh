#!/bin/bash

DOCKERTAG=energy:2.0
GIT_ROOT=`git rev-parse --show-toplevel`

NETWORK=220124cost5input_10_0_20
CONFIG=config

docker run -it \
-e BASE64_ENCODED=false \
-v ${GIT_ROOT}/input/${NETWORK}.json:/var/input/data/data.json \
-v ${GIT_ROOT}/input/${CONFIG}.json:/var/input/params/params.json \
${DOCKERTAG}  | tee  >(tail -n 1 > JobResponse)

exit 0
