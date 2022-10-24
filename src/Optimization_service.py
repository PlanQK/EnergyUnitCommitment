import streamlit as st
import time

import pandas as pd
import altair as alt

import requests

import json
import yaml
import os

from json import JSONDecodeError

from contextlib import redirect_stdout
import io

from ast import literal_eval

st.set_page_config(page_title="Unit Commitment Optimization", layout="wide")
state = st.session_state


if "init" not in state:
    state.init = True
    state.network = None
    state.config = {}
    state.buses = None
    state.result = None
    state.result_dict = {}
    state.logs = ""
    state.optimization_state = "waiting"
    state.delete = False
    state.current_path = [None]
    state.current_scheme_index = None
    state.config_as_string = ""
    state.config_id = None
    state.format_yaml = True
    state.format_json = False

@st.cache(show_spinner=False)
def get_url():
    url = "http://localhost:443"
    if os.path.isfile("/url.txt"):
        with open("/url.txt", "r") as f:
            url = f.readline().strip()
    elif os.path.isfile("./url.txt"):
        with open("./url.txt", "r") as f:
            url = f.readline().strip()
    return url

url = get_url()

# Process and display result data
@st.cache(show_spinner=False)
def create_generator_frame(result):
    return transform_result_to_frame(result["results"]["unit_commitment"], 
                                     get_network_generators(),
                                     component_column_name="generator")

@st.cache(show_spinner=False)
def create_line_frame(result):
    return transform_result_to_frame(result["results"]["powerflow"], 
                                     get_network_lines(),
                                     component_column_name="transmission line")

@st.cache(show_spinner=False)
def create_kirchhoff_frame(result):
    # add summed kirchhoff penalites of all buses to the indivudual penalty
    # of each bus
    complete_network_penalty = result["results"]["kirchhoff_cost_by_time"]
    complete_network_penalty = {
                                str(("total", cast_snapshot(key))): value 
                                for key, value in complete_network_penalty.items()
                             }
    return transform_result_to_frame({**result["results"]["individual_kirchhoff_cost"],
                                      **complete_network_penalty},
                                     get_buses()  + ["total"],
                                     component_column_name="bus")

def transform_result_to_frame(result_dict, network_components, component_column_name="component"):
    data_table = []
    snapshots = get_network_snapshots()
    for component in network_components:
        keys = [str((component, snapshot)) for snapshot in snapshots]
        try:
            df_row = [component] + [result_dict[key] for key in keys]
        except KeyError:
            keys = [f"('{component}', Timestamp('{snapshot}'))" for snapshot in snapshots]
            df_row = [component] + [result_dict[key] for key in keys]
        data_table.append(df_row)
    df = pd.DataFrame(data_table, columns=[component_column_name] + list(snapshots))
    df = df.set_index(df.columns[0])
    return df

def visualize_component_solution(component_list, 
                                 data_frame,
                                 value_key,
                                 select_message="Choose generators",
                                 component_column_name="component"):
    options = st.multiselect(
        select_message,
        component_list,
        component_list,
        )
    if options:
        df = data_frame.filter(items = options, axis=0).reindex(options).transpose().reset_index().melt('index')
        df.columns = ["snapshot",component_column_name, value_key]
        chart = alt.Chart(df).mark_line().encode(
            alt.X('snapshot:N',axis=alt.Axis(tickMinStep=1)),
            y=value_key,
            color=component_column_name,
            )
        st.altair_chart(chart, use_container_width=True)

@st.cache(show_spinner=False)
def get_network(response, upload_id):
    response_dict = response.json()
    get_network_info(response_dict)
    return response_dict['filename']

def get_network_generators():
    return state.generators

def get_network_lines():
    return state.lines

def get_buses():
    return state.buses

def cast_snapshot(key):
    try:
        return int(key)
    except:
        return key

def get_network_info(post_response):
    state.generators = post_response['generators']
    state.lines = post_response['lines']
    state.snapshots = post_response['snapshots']
    state.buses = post_response['buses']

def get_network_snapshots():
    return [cast_snapshot(snapshot) for snapshot in state.snapshots][:state.config.get("snapshots", None)]

def set_config_format(string):
    if string == 'json':
        state.format_yaml = not state.format_json
    elif string == 'yaml':
        state.format_json = not state.format_yaml

def dump_converted_config(config=None):
    if config is None:
        config = state.config
    if state.config:
        if state.format_yaml:
            state.config_as_string = yaml.dump(config, default_flow_style=False, indent=4)
        elif state.format_json:
            state.config_as_string = json.dumps(config, indent=4)
    else:
        state.config_as_string = ""
    return state.config_as_string

def update_config(config=None):
    if config is None:
        config = state.config_as_string
    try:
        new_value = yaml.safe_load(config)
        if isinstance(new_value, dict):
            state.config = new_value
    except:
        pass
    dump_converted_config(state.config)

def delete_configuration():
    state.config_as_string = ""
    state.config = {}


"""
# Optimization service of the unit commitment problem

This webservice allows you to upload [PyPSA network](https://pypsa.readthedocs.io/en/latest/)
network files and run an optimization of it's unit commitment problem using quantum or classical methods
"""

col1, col2 = st.columns(2)
with col1:
    network = st.file_uploader("Choose a network file", ["nc"],)
with col2:
    config = st.file_uploader("Choose a configuration file", ["yaml", "yml", "json"])


col1, col2, col3, col4 = st.columns([7,2,2,10])
with col1:
    "Displayed format of configuration:"
with col2:
    st.checkbox("json", key="format_json", on_change=set_config_format, args=('json',))
with col3:
    st.checkbox("yaml", key="format_yaml", on_change=set_config_format, args=('yaml',))

# Obtain config data and write into state
if config is not None:
    if config.id != state.config_id:
        state.config_id = config.id
        update_config(config)
dump_converted_config()

# Editor for configuration files
col1, col2 = st.columns([8,2])
with col1:
    placeholder = "You can either paste and edit a configuration here or uploaded it via the upload widget"
    json_text = st.text_area("Edit the configuration JSON",    
                            key="config_as_string",
                            on_change=update_config,
                            height=200,
                            label_visibility='collapsed',
                            placeholder=placeholder
                            )
with col2:
    st.button("Reupload configuration file", 
              on_click=update_config, 
              args=(config,),
              disabled=not bool(config))
    st.button("Delete configuration", 
              on_click=delete_configuration,
              disabled=not bool(state.config))
    if state.format_yaml:
        filename = "configuration.yml"
    elif state.format_json:
        filename = "configuration.json"
    st.download_button("Download configuration", data=state.config_as_string, file_name=filename)

if network is None:
    state.result = None
    state.logs = None
    st.write("Waiting for an uploaded network")
else:
    starter = st.button(label="Run optimization")
    if starter:
        files = {
            'network': network.getvalue()
        }
        upload_post = requests.request("POST", 
                                       url + "upload_network",
                                       files=files,
                                       headers={},
                                       data={})
        get_network(upload_post, network.id)
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            "network": "network.nc",
            "config": state.config
        }
        response = requests.request("POST", url + "start", headers=headers, data=json.dumps(payload))
        state.result = response.json()
        state.logs = state.result['logs']
        state.result_dict = json.loads(state.result['result'])['result']
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
            st.json(state.result['result'])
    with col2:
        st.download_button("Download JSON", data=state.result['result'], file_name="result.json")

    # Extract and display result data
    try:
        generator_frame = create_generator_frame(state.result_dict)
        line_frame = create_line_frame(state.result_dict)
        kirchhoff_frame = create_kirchhoff_frame(state.result_dict)
    except KeyError as e:
        "Optimization yielded no feasible result"
    if 'line_frame' in locals():
        with st.expander("Network component selection", expanded=True):
            col1, col2 = st.columns([1,1])
            with col1:
                visualize_component_solution(get_network_generators(),
                                            generator_frame,
                                            'power level',
                                            select_message="Choose generators",
                                            component_column_name="generator")
            with col2:
                visualize_component_solution(get_network_lines(),
                                            line_frame,
                                            'powerflow',
                                            select_message="Choose transmission lines",
                                            component_column_name="transmission line")

        with st.expander("Kirchhoff cost visualization", expanded=True):
            visualize_component_solution(list(kirchhoff_frame.index),
                                        kirchhoff_frame,
                                        'kirchhoff penalty',
                                        select_message="Choose buses",
                                        component_column_name="bus")
