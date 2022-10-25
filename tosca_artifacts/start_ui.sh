#!/bin/bash
echo "starting ui"
cd /EnergyUnitCommitment/
nohup streamlit run Optimization_service.py --server.port 80 &