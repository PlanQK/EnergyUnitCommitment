"""
this small script converts configuration yamls and network nc files into
json. In order to run, you have to install pypsa and PyYAML
"""

import yaml
import json
import pypsa

import sys
import os


def main():
    # make a prefix so all paths start from git root. 
    if os.path.split(__file__)[0] == 'input':
        path_prefix = "./"
    else:
        path_prefix = "../"

    input_file = sys.argv[1]
    filetype = input_file.split(".")[-1]
    output_file = path_prefix + "input/" + sys.argv[2] + ".json"

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
