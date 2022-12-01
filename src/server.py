"""Starts a Flask server that can provided a REST API to the streamlit frontend
for Unit Commitment optimziation"""

from distutils.command.config import config
from tokenize import String

from contextlib import redirect_stdout
import io
import json
import os

from werkzeug.utils import secure_filename

import pypsa
from flask import Flask
from flask import request


from run import parse_cli_params
from program import run

app = Flask(__name__)


def get_root_path():
    if os.environ.get('CONTAINERLESS', False):
        path = os.getcwd() + '/input/networks/'
    else:
        path = '/EnergyUnitCommitment/networks/'
    return path

@app.route('/start', methods=['POST'])
def start_optimization():

    data = request.get_json()

    config = data['config']
    network = data['network']
    if isinstance(network, str):
        path = get_root_path() + secure_filename(network)
        network = pypsa.Network(path)

    f = io.StringIO()
    if isinstance(config, str):
        config=json.loads(config)
    with redirect_stdout(f):
        result = run(data=network, params=config)
    response = {
        'result': result.to_json(),
        'logs': f.getvalue()
    } 
    return json.dumps(response)

@app.route('/upload_network', methods=['POST'])
def upload_network():
    file = request.files['network']

    path = get_root_path() + secure_filename(file.filename) + ".nc"

    with open(path, 'wb', encoding='utf-8') as f:
        f.write(file.getvalue())

    network = pypsa.Network(path)
    response = {
            'filename': file.filename,
            'generators': list(network.generators.index),
            'lines': list(network.lines.index),
            'snapshots': [str(snapshot) for snapshot in network.snapshots],
            'buses': list(network.buses.index)
            }
    return json.dumps(response)

port = os.environ.get('Port', 443)
os.makedirs(get_root_path(), exist_ok=True)

app.run(host='0.0.0.0', port=port)
