"""
this small script converts configuration yamls and network nc files into
json. In order to run, you have to install pypsa and PyYAML
"""

import yaml
import json
import pypsa

import sys
import os

usage_string = """
This takes up to two arguments, one for the name of the file to be loaded, and one
for the name of the file the json gets dumped to. Ommitting the last parameter will
set the save file name to the same name as the input but changing it to json

networks are assumed to be in the folder sweetNetworks/
config files are assumed to be in the folder src/Configs/

convert_to_json some_network_or_config_with_ending save_file_without_json
                some_network_or_config_with_ending
"""

def main():
    if len(sys.argv) == 1:
        print(usage_string)
        return
    
    # make a prefix so all paths start from git root. 
    if os.path.split(__file__)[0] == 'input':
        path_prefix = "../"
    else:
        path_prefix = "./"

    input_file = sys.argv[1]
    filetype = input_file.split(".")[-1]
    if len(sys.argv) >= 3:
        output_name = sys.argv[2]
    else:
        output_name = input_file[: -len(filetype) - 1]

    output_file = path_prefix + "input/" + output_name + ".json"

    print(f"reading {input_file}")
    if filetype == "yaml":
        with open(path_prefix + "src/Configs/" + input_file, 'r') as yaml_in, open(output_file, "w") as json_out:
            yaml_object = yaml.safe_load(yaml_in)  # yaml_object will be a list or a dict
            json.dump(yaml_object, json_out, indent=2)
    if filetype == "nc":
        print(f"reading network and writing as json")
        network = pypsa.Network(path_prefix + "sweepNetworks/" + input_file)
        networkx_array = network.export_to_netcdf()
        with open(output_file, "w") as write_file:
            json.dump(networkx_array.to_dict(), write_file, indent=2)
    print(f"writing {output_file} is done")


if __name__ == "__main__":
    main()
