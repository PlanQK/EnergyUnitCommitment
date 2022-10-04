from distutils.command.config import config
from flask import Flask
from flask import request
from run import parse_cli_params
from program import run

app = Flask(__name__)

@app.route('/start', methods=['POST'])
def start_optimization():
    #network = request.args['network']
    #config = request.args['config']
    #cli_params_dict = parse_cli_params(request.args.get('cli_params', ""))
    network = request.json["data"]
    config = request.json["params"]
    response = run(data=network, params=config)
    return response.to_json()

app.run(host='0.0.0.0', port=80)