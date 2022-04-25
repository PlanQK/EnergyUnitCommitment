import json

import xarray
import yaml
import pypsa

from typing import Union


class InputReader:
    """
    This class is an reader to obtain the configuration dictionary dependent on the input format
    """
    def __init__(self, network: Union[pypsa.Network, str], config: Union[dict, str]):
        self.network, self.networkName = self.makeNetwork(network)
        self.config = self.makeConfig(config)

    def makeNetwork(self, network: Union[str, dict, pypsa.Network]) -> [pypsa.Network, str]:
        if isinstance(network, str):
            return pypsa.Network(f"Problemset/" + network), network
        if isinstance(network, dict):
            loadedDataset = xarray.Dataset.from_dict(network)
            loadedNet = pypsa.Network(name="")
            pypsa.Network.import_from_netcdf(network=loadedNet, path=loadedDataset)
            return loadedNet, "network_from_dict"
        if isinstance(network, pypsa.Network):
            return network, "no_name_network"
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

    def getNetworkName(self) -> str:
        return self.networkName

