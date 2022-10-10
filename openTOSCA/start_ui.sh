#!/bin/bash
echo "starting ui"
ls -la /EnergyUnitCommitment
cd /EnergyUnitCommitment/
nohup streamlit run Optimization_service.py --server.port 80 &