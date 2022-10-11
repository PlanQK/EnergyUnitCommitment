import streamlit as st
import time

from program import run

import pypsa
import pandas as pd
import altair as alt

import json
import yaml

from contextlib import redirect_stdout
import io

from ast import literal_eval

st.set_page_config(page_title="Unit Commitment Optimization", layout="wide")
state = st.session_state


# Process and display result data
def transform_result_to_frame(result_dict, network_components):
    data_table = []
    snapshots = state.network.snapshots
    for component in network_components:
        keys = [str((component, snapshot)) for snapshot in snapshots]
        df_row = [component] + [result_dict[key] for key in keys]
        data_table.append(df_row)
    df = pd.DataFrame(data_table, columns=["component"] + list(snapshots))
    df = df.set_index(df.columns[0])
    return df

# @st.cache
def create_generator_frame(result):
    return transform_result_to_frame(result["results"]["unit_commitment"], state.network.generators.index)

# @st.cache
def create_line_frame(result):
    return transform_result_to_frame(result["results"]["powerflow"], state.network.lines.index)

def visualize_component_solution(component_list, data_frame, value_key):
    options = st.multiselect(
        "Choose generators",
        component_list,
        component_list,
        )
    if options:
        df = data_frame.filter(items = options, axis=0).reindex(options).transpose().reset_index().melt('index')
        df.columns = ["snapshot", "component", value_key]
        chart = alt.Chart(df).mark_line().encode(
            alt.X('snapshot:N',axis=alt.Axis(tickMinStep=1)),
            y=value_key,
            color='component',
            )
        st.altair_chart(chart, use_container_width=True)



if "init" not in state:
    state.init = True
    state.network = None
    state.config = {}
    state.result = None
    state.result_dict = {}
    state.logs = ""
    state.optimization_state = "waiting"
    state.delete = False
    state.current_path = [None]
    state.current_scheme_index = None

# @st.cache
def get_network(network_file):
    with open("hack_file.nc", 'wb') as f:
        f.write(network_file.getvalue())
    state.network = pypsa.Network("./hack_file.nc")

# @st.cache
def get_config(config):
    state.config = yaml.safe_load(config)

def get_network_generators():
    return list(state.network.generators.index)

def get_network_lines():
    return list(state.network.lines.index)

"""
# Optimization service of the unit commitment problem

This webservice allows you to upload [PyPSA network](https://pypsa.readthedocs.io/en/latest/)
network files and run an optimization of it's unit commitment problem using quantum or classical methods
"""

col1, col2 = st.columns(2)

with col1:
    network = st.file_uploader("Choose a network file", ["nc"],)
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
    starter = st.button(label="Run optimization")
    if starter:
        with st.spinner('Wait for it...'):
            f = io.StringIO()
            with redirect_stdout(f):
                response = run(data=state.network, params=state.config)
            state.logs = f.getvalue()
            state.result = response.to_json()
            state.result_dict = response.result
            st.success('Done!')


# Download results
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

    # Extract and display result data
    try:
        generator_frame = create_generator_frame(state.result_dict)
        line_frame = create_line_frame(state.result_dict)
    except KeyError:
        "Optimization yielded no feasible result"
    if 'line_frame' in locals():
        with st.expander("Network component selection", expanded=True):
            col1, col2 = st.columns([1,1])
            with col1:
                visualize_component_solution(get_network_generators(),
                                            generator_frame,
                                            'power level')
            with col2:
                visualize_component_solution(get_network_lines(),
                                            line_frame,
                                            'powerflow')

