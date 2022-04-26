#!/bin/bash

git_root=`git rev-parse --show-toplevel`
path_prefix=""
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
params=`cat $file | jq '.params' | base64`

echo "jq done"

winpty docker run -it -e INPUT_DATA="${data}" -e INPUT_PARAMS="${params}" planqk-service


