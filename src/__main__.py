import json

import yaml

from program import run

input_file = 'testNetwork4QubitIsing.json'
input_config = "config.yaml"
with open(f"Configs/{input_file}") as file:
    inp = json.load(file)

with open(f"Configs/{input_config}") as file:
    conf = yaml.safe_load(file)

data = inp
params = conf

response = run(data, params)

response.save_to_json_local_docker(folder="../results_test_sweep/")
#print()
#print(response.to_json())
