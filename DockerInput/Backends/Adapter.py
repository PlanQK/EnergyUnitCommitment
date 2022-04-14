import json
import yaml
import pypsa

import typing
from typing import Union

class Adapter:
    """
    This class is an adapter to obtain the configuration dictionary dependent on the input format
    """
    def __init__(self, network: Union[pypsa.Network, str] , params: Union[dict, str]):
        self.network = self.makeNetwork(network)
        self.config = self.makeConfig(params)


    def makeNetwork(self, network :Union[str, dict, pypsa.Network]) -> pypsa.Network:
        if isinstance(network, str):
            return pypsa.Network(f"Problemset/" + network)
        if isinstance(network, dict):
            raise NotImplementedError
        if isinstance(network, pypsa.Network):
            return network
        raise NotImplementedError
    
    def makeConfig(self, params: Union[dict, str]) -> dict:
        if isinstance(params, dict):
            return params
        if isinstance(params, str):
            filetype = params[-4:]
            if filetype == "json":
                with open(params) as file:
                    return json.load(file)
            elif filetype == "yaml":
                with open("Configs/" + params) as file:
                    return yaml.safe_load(file)
        raise ValueError("input can't be read")


    def getConfig(self) -> dict:
        return self.config

    def getNetwork(self) -> pypsa.Network:
        return self.network

