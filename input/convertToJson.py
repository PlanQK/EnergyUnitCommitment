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
        pathPrefix = "./"
    else:
        pathPrefix = "../"

    inputFile = sys.argv[1]
    filetype = inputFile.split(".")[-1]
    outputFile = pathPrefix + "input/" + sys.argv[2] + ".json"

    print(f"reading {inputFile}")
    if filetype == "yaml":
        with open(pathPrefix + "src/Configs/"+ inputFile, 'r') as yaml_in, open(outputFile, "w") as json_out:
            yaml_object = yaml.safe_load(yaml_in) # yaml_object will be a list or a dict
            json.dump(yaml_object, json_out, indent=2)
    if filetype == "nc":
        print(f"reading network and writing as json")
        network = pypsa.Network(pathPrefix + "sweepNetworks/" + inputFile)
        networkXarray = network.export_to_netcdf()
        with open(outputFile, "w") as write_file:
            json.dump(networkXarray.to_dict(), write_file, indent=2)
    print(f"writing {outputFile} is done")

if __name__ == "__main__":
    main()
