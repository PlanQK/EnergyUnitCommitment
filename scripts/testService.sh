#!/bin/bash

git_root=`git rev-parse --show-toplevel`
path_prefix="/input/"
path_to_json="testService"

file="${git_root}/${path_prefix}${path_to_json}.json"

while [ ! -z "$1" ];
do
    case $1 in

        -b)
            docker build -t planqk-service $git_root
            ;;

        *)
            file="${git_root}/${path_prefix}${1}.json"
            ;;
    esac
    shift
done

echo "using ${file} as input"

data=`cat $file | jq '.data' | base64`
inputConfig=`cat $file | jq '.inputConfig' | base64`

echo "jq done"

unameOut=$(uname -s)

case $unameOut in
    Linux)
        docker run -it -e INPUT_DATA="${data}" -e INPUT_PARAMS="${inputConfig}" planqk-service
        ;;
    *)
        winpty docker run -it -e INPUT_DATA="${data}" -e INPUT_PARAMS="${inputConfig}" planqk-service
        ;;
esac


