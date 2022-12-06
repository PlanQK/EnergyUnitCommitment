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
  mkdir /EnergyUnitCommitment
  if [[ "${entry[1]}" == *.py ]];
  then
    # copy the executable to /qhana_backend
	cp $CSAR${entry[1]} /EnergyUnitCommitment/
  fi
done
echo "Finished Unpacking"
echo "Installing Dependencies"
pip install streamlit
pip install PyYAML