#!/bin/bash

GIT_ROOT=`git rev-parse --show-toplevel`

. ${GIT_ROOT}/venv/bin/activate

if [ -z "${Port}" ]; then
    echo "No Port passed in environment, defaulting to 1443"
    export Port=1443
fi

export CONTAINERLESS=1
export TRUSTED_USER=Yes

echo "http://localhost:${Port}/" > ${GIT_ROOT}/url.txt

trap 'trap " " SIGTERM; kill 0; wait; echo "Shutdown of flask and streamlit complete"' SIGINT SIGTERM

cd ${GIT_ROOT} && python3 ./src/server.py &
cd ${GIT_ROOT} && streamlit run ./src/Optimization_service.py &

wait
