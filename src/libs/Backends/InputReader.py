import json

import xarray
import yaml
import pypsa

import copy

from typing import Union


class InputReader:
    """
    This class is an reader to obtain the configuration dictionary dependent on the input format
    """
    BackendToSolver = {
            "DWaveBackend": ["dwave-tabu", "dwave-greedy", "dwave-hybrid", "dwave-qpu", "dwave-read-qpu"],
            "PypsaBackend": ["pypsa-glpk", "pypsa-fico"],
            "SqaBackend": ["sqa", "classical"],
            "QaoaBackend": ["qaoa"],
    }

    def __init__(self, 
                network: Union[pypsa.Network, str],
                config: Union[dict, str],
                extraParams: list = []):
        self.network, self.networkName = self.makeNetwork(network)
        self.config = self.makeConfig(config)
        self.addExtraParameters(extraParams)
        self.copyToBackendConfig()
        # print final config, but hide api tokens
        configWithoutToken = copy.deepcopy(self.config)
        for provider, token in self.config["APItoken"].items():
            if token != '':
                configWithoutToken["APItoken"][provider] = "*****"
        print(f"running with the following configuration {configWithoutToken}")

    def copyToBackendConfig(self):
        for BackendType, solverList in self.BackendToSolver.items():
            if self.config["Backend"] in solverList:
                sourceKey = BackendType
                break
        self.config["BackendConfig"] = {**self.config["BackendConfig"],**self.config[sourceKey]}

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
        result = {}
        if isinstance(params, dict):
            result = params
        if isinstance(params, str):
            filetype = params[-4:]
            if filetype == "json":
                with open(params) as file:
                    result = json.load(file)
            elif filetype == "yaml":
                with open("Configs/" + params) as file:
                    result = yaml.safe_load(file)
        if result["BackendConfig"] is None:
            result["BackendConfig"] = {}
        return result

    def addExtraParameters(self, extraParams: list):
        for keyChain in extraParams:
            descentInConfig = self.config
            for key in keyChain[:-2]:
                try:
                    descentInConfig = descentInConfig[key]
                except KeyError:
                    descentInConfig[key] = {}
                    descentInConfig = descentInConfig[key]
            descentInConfig[keyChain[-2]] = keyChain[-1]
    
    def getConfig(self) -> dict:
        return self.config

    def getNetwork(self) -> pypsa.Network:
        return self.network

    def getNetworkName(self) -> str:
        return self.networkName

