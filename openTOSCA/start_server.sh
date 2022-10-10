#!/bin/bash
echo "starting server"
ls -la /EnergyUnitCommitment
cd /EnergyUnitCommitment/src
nohup python server.py &