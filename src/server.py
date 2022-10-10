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

@app.route('/start', methods=['POST'])
def start_optimization():
    #network = request.args['network']
    #config = request.args['config']
    #cli_params_dict = parse_cli_params(request.args.get('cli_params', ""))

    data = request.get_json()
    

    config = data['config']
    network = data['network']
    if isinstance(network, str):
        path = '/EnergyUnitCommitment/networks/' + secure_filename(network)
        network = pypsa.Network(path)

    f = io.StringIO()
    with redirect_stdout(f):
        result = run(data=network, params=config)
    response = {
        'result':result.to_json(),
        'logs':f.getvalue()
    } 
    return json.dumps(response)

@app.route('/upload_network', methods=['POST'])
def upload_network():
    file = request.files['network']
    path = '/EnergyUnitCommitment/networks/' + secure_filename(file.filename)
    file.save(path)
    return file.filename

port = os.environ['Port']

app.run(host='0.0.0.0', port=port)