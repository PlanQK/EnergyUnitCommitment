"""
This small script converts configuration yaml's and network netcdf4 files into
json. In order to run, you have to install pypsa and PyYAML
"""

import json
import sys
import os
import pathlib

import yaml
import pypsa



usage_string = """
This takes up to two arguments, one for the name of the file to be loaded, and one
for the name of the file the json gets dumped to. Omitting the last parameter will
set the save file name to the same name as the input but changing it to json

networks are assumed to be in the folder input/networks/
config files are assumed to be in the folder input/configs/

convert_to_json some_network_or_config_with_ending save_file_without_json
                some_network_or_config_with_ending
"""


def set_path_prefix():
    """
    set path to the unit commitment repo as global variable, so you can call
    this script from anywhere and it still finds your network and config folder
    """
    global path_prefix
    path_prefix = pathlib.Path(os.path.split(__file__)[0]).parent
    print(f"Current path to repo is: {path_prefix}")


def convert_network(input_name: str, output_name: str):
    """
    Takes the name of a network in `networks`, converts it 
    to json, and dumps it to `input` using the output_name
    
    Args:
        input_name: (str)
            the name of the input network without the file extension
        output_name: (str)
            the name out the output json without the file extension
    """
    input_path = os.path.join(path_prefix, "input/networks", input_name) + ".nc"
    output_path = os.path.join(path_prefix, "input", output_name) + ".json"

    print("reading network")
    network = pypsa.Network(input_path)
    print("write converted file")
    network_xarray = network.export_to_netcdf()
    with open(output_path, "w", encoding='utf-8') as write_file:
        json.dump(network_xarray.to_dict(), write_file, indent=2)


def convert_config(input_name: str, output_name: str):
    """
    Takes the name of a config file in yaml format in `configs`, converts it 
    to json, and dumps it to `input` using the output_name

    Args:
        input_name: (str)
            the name of the input configuration without the file extension
        output_name: (str)
            the name out the output json without the file extension
    """
    input_path = os.path.join(path_prefix, "input/configs", input_name) + ".yaml"
    output_path = os.path.join(path_prefix, "input", output_name) + ".json"

    with open(input_path, 'r', encoding='utf-8') as yaml_input,  open(output_path, "w", encoding='utf-8') as json_out:
        print("read configuration")
        yaml_object = yaml.safe_load(yaml_input)
        print("write converted file")
        json.dump(yaml_object, json_out, indent=2)


def main():
    if len(sys.argv) == 1:
        print(usage_string)
        return

    # setup file names
    set_path_prefix()
    input_name, file_extension = os.path.splitext(sys.argv[1])
    output_name = input_name
    if len(sys.argv) >= 3:
        output_name = sys.argv[2]

    # read and write
    if file_extension == ".yaml":
        convert_config(input_name, output_name)
    elif file_extension == ".nc":
        convert_network(input_name, output_name)
    print("done")


if __name__ == "__main__":
    main()
