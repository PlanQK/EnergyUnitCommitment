from distutils.command.config import config
from tokenize import String
from flask import Flask
from flask import request
from run import parse_cli_params
from program import run
from werkzeug.utils import secure_filename
from contextlib import redirect_stdout
import io
import json
import pypsa
import os

app = Flask(__name__)


def get_root_path():
    if os.environ.get('CONTAINERLESS', False):
        path = os.getcwd() + '/networks/' 
    else:
        path = '/EnergyUnitCommitment/networks/' 
    return path

@app.route('/start', methods=['POST'])
def start_optimization():
    #network = request.args['network']
    #config = request.args['config']
    #cli_params_dict = parse_cli_params(request.args.get('cli_params', ""))

    data = request.get_json()

    config = data['config']
    network = data['network']
    if isinstance(network, str):
        path = get_root_path() + secure_filename(network)
        network = pypsa.Network(path)

    print("CONG")
    print(config)
    
    f = io.StringIO()
    if isinstance(config, str):
        config=json.loads(config)
    with redirect_stdout(f):
        result = run(data=network, params=config)
    print(result.to_json())
    response = {
        'result': result.to_json(),
        'logs': f.getvalue()
    } 
    return json.dumps(response)

@app.route('/upload_network', methods=['POST'])
def upload_network():
    file = request.files['network']

    path = get_root_path() + secure_filename(file.filename) + ".nc"

    with open(path, 'wb') as f:
        f.write(file.getvalue())

    network = pypsa.Network("./networks/network.nc")
    response = {
            'filename': file.filename,
            'generators': list(network.generators.index),
            'lines': list(network.lines.index),
            'snapshots': list(network.snapshots)
            }
    return json.dumps(response)

port = os.environ['Port']

app.run(host='0.0.0.0', port=port)
