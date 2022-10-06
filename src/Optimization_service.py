import streamlit as st
import time

from program import run

import pypsa

import json
import yaml

from contextlib import redirect_stdout
import io


st.set_page_config(layout="wide")
state = st.session_state


if "init" not in state:
    state.init = True
    state.network = None
    state.config = {}
    state.result = None
    state.result_dict = {}
    state.logs = ""

def get_network(network_file):
    with open("hack_file.nc", 'wb') as f:
        f.write(network.getvalue())
    state.network = pypsa.Network("./hack_file.nc")

def get_config(config):
    state.config = yaml.safe_load(config)

"""
# Optimization service of the unit commitment problem

This webservice allows you to upload [PyPSA network](https://pypsa.readthedocs.io/en/latest/)
network files and run an optimization of it's unit commitment problem using quantum or classical methods
"""


col1, col2 = st.columns(2)

with col1:
    network = st.file_uploader("Choose a network file", ["nc", "json"],)
with col2:
    config = st.file_uploader("Choose a configuration file", ["yaml", "json"])


if config is not None:
    get_config(config)
else:
    state.config = {}


if network is None:
    state.result = None
    state.logs = None
    st.write("Waiting for an uploaded network")
else:
    get_network(network)
    if state.result is None:
        button_label = "Run Optimization"
    else:
        button_label = "Rerun Optimization"
    starter = st.button(button_label)
    if starter:

        with st.spinner('Wait for it...'):
            f = io.StringIO()
            with redirect_stdout(f):
                response = run(data=state.network, params=state.config)
            state.logs = f.getvalue()
            state.result = response.to_json()
            state.result_dict = response.result
            st.success('Done!')

if state.result is not None:
    col1, col2 = st.columns([8,1])
    with col1:
        with st.expander("CLI Logs"):
            st.code(state.logs, language=None)
    with col2:
        st.download_button("Download Logs", data=state.logs, file_name="optimization_logs")

    col1, col2 = st.columns([8,1])
    with col1:
        with st.expander("JSON Result"):
            st.json(state.result)
    with col2:
        st.download_button("Download JSON", data=state.result, file_name="result.json")

    if len(state.network.snapshots) > 1:
        generator = st.selectbox("Choose a generator", list(state.network.generators.index))
        snapshots = state.network.snapshots
        unit_commitment = state.result_dict["results"]["unit_commitment"]
        x_axis = [str((generator, snapshot)) for snapshot in snapshots]
        y_axis = [unit_commitment[key] for key in x_axis]
        st.line_chart(y_axis)

