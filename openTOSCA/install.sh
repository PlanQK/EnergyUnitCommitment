#!/bin/bash

echo "Unpacking Archiv"
IFS=';' read -ra NAMES <<< "$DAs";
for i in "${NAMES[@]}"; do
  echo "KeyValue-Pair: "
  echo $i
  IFS=',' read -ra entry <<< "$i";
    echo "Key: "
    echo ${entry[0]}
    echo "Value: "
    echo ${entry[1]}

  # find the executable jar file
  if [[ "${entry[1]}" == *.tar.gz ]];
  then
    # copy the executable to /qhana_backend
	tar -xf $CSAR${entry[1]} -C /
  fi
done
echo "Finished Unpacking"
echo "Installing Dependencies"

pip install pyomo==6.4.1 && \
pip install pypsa==0.20.0 && \
pip install numpy==1.23.1 && \
pip install scipy==1.8.1 && \
pip install pandas==1.4.3 && \
pip install matplotlib==3.5.2 
pip install dimod==0.11.2 && \
pip install qiskit==0.36.2 && \
pip install networkx==2.8.4 && \
pip install tables==3.7.0 && \
pip install minorminer==0.2.9

cd EnergyUnitCommitment
pip install -r requirements.txt
apt install -y glpk-utils

echo "Finished Installing Dependencies"
export RUNNING_IN_DOCKER=Yes
export TRUSTED_USER=Yes
mkdir input-model