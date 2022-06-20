import copy
import json
import pypsa
import xarray
import yaml

from typing import Union


class InputReader:
    """
    This class is a reader to obtain the configuration dictionary and
    pypsa.Network, dependent on the input format.
    """

    # dictionary which solver has which backend specific extra data where.
    # Keys are broader categories of backends and values are a list of solvers
    # that use that key to store additional config info.
    BackendToSolver = {
        "DWaveBackend": [
            "dwave-tabu",
            "dwave-greedy",
            "dwave-hybrid",
            "dwave-qpu",
            "dwave-read-qpu",
        ],
        "PypsaBackend": ["pypsa-glpk", "pypsa-fico"],
        "SqaBackend": ["sqa", "classical"],
        "QaoaBackend": ["qaoa"],
    }
    loaders = {
        "json": json.load,
        "yaml": yaml.safe_load
    }

    def __init__(self,
                 network: Union[str, dict, pypsa.Network],
                 config: Union[str, dict],
                 extraParams: list = [],
                 extraParamValues: list = []
                 ):
        """
        Obtain the configuration dictionary and pypsa.Network,
        dependent on the input format of network and config. The
        obtained configuration dictionary is then amended with the
        extraParams, if provided.

        Args:
            network: (Union[str, dict, pypsa.Network])
                A string, dictionary or pypsa.Network, representing the
                pypsa.Network to be used.
            config: (Union[str, dict])
                A string or dictionary to be used to obtain the
                configuration for the problem instances.
            extraParams: (list)
                A list of the extra parameters to be added to the
                config dictionary.
        """
        self.network, self.networkName = self.makeNetwork(network)
        self.config = self.makeConfig(config)
        self.addExtraParameters(extraParams=extraParams,
                                extraParamValues=extraParamValues)
        self.copyToBackendConfig()
        # print final config, but hide api tokens
        configWithoutToken = copy.deepcopy(self.config)
        for provider, token in self.config["APItoken"].items():
            if token != "":
                configWithoutToken["APItoken"][provider] = "*****"
        print(f"running with the following configuration {configWithoutToken}")

    def copyToBackendConfig(self) -> None:
        """
        Copies the Backend specific configuration to the Backend
        agnostic key "BackendConfig" where it is merged with any already
        present dictionary entries.

        Returns:
            (None)
                Modifies self.config.
        """
        self.config["Backend"] = self.config["Backend"].replace('_', '-')
        for BackendType, solverList in self.BackendToSolver.items():
            if self.config["Backend"] in solverList:
                self.config["BackendConfig"] = {
                    **self.config.get("BackendConfig", {}),
                    **self.config[BackendType],
                }
                return

    def makeNetwork(self, network: Union[str, dict, pypsa.Network]) \
            -> [pypsa.Network, str]:
        """
        Opens a pypsa.Network using the provided network argument. If a
        string is given it is interpreted as the network name. It has to
        be stored in the folder "Problemset". If a dictionary is given,
        it will be assumed to be the dictionary representation of a
        netCDF format and will be converted into a pypsa.Network.
        Lastly, if a pypsa.Network is provided it will just be passed
        through to the output.

        Args:
            network: (Union[str, dict, pypsa.Network])
                A string, dictionary or pypsa.Network, representing the
                pypsa.Network to be used.

        Returns:
            (pypsa.Network)
                The network to be used in the problem instances.
            (str)
                The network name.
        """
        if isinstance(network, str):
            return pypsa.Network(f"Problemset/" + network), network
        elif isinstance(network, dict):
            loadedDataset = xarray.Dataset.from_dict(network)
            loadedNet = pypsa.Network(name="")
            pypsa.Network.import_from_netcdf(network=loadedNet,
                                             path=loadedDataset)
            return loadedNet, "network_from_dict"
        elif isinstance(network, pypsa.Network):
            return network, "no_name_network"
        raise TypeError("The network has to be given as a dictionary, "
                        "representing the netCDF format of a pypsa.Network, "
                        "an actual pypsa.Network, or a string with the name "
                        "of the pypsa.Network, which has to be stored in the "
                        "Problemset folder.")

    def makeConfig(self, inputConfig: Union[str, dict]) -> dict:
        """
        Converts an inputConfig file into a dictionary. If the input is
        a dictionary it will be passed through to the output. If it is
        a string, the filetype will be determined, the file opened and
        stored in a dictionary. Currently .json and .yaml files are
        supported. Before being returned, it is checked if the key
        "BackendConfig" is present, if not it will be created.

        Args:
            inputConfig: (Union[str, dict])
                A string or dictionary to be used to obtain the
                configuration for the problem instances.

        Returns:
            (dict)
                The config stored in a dictionary.
        """
        if isinstance(inputConfig, dict):
            result = inputConfig
        else:
            try:
                # inputConfig is assumed to be the path if it is not a dict
                filetype = inputConfig.split(".")[-1]
                loader = self.loaders[filetype]
            except KeyError:
                raise KeyError(f"The file format {filetype} doesn't match any supported "
                               f"format. The supported formats are {list(self.loaders.keys())}")
            with open("Configs/" + inputConfig) as file:
                result = loader(file)
        if not isinstance(result.get("BackendConfig", None), dict):
            result["BackendConfig"] = {}
        return result

    def addExtraParameters(self,
                           extraParams: list,
                           extraParamValues: list
                           ) -> None:
        """
        Writes extra parameters into the config dictionary, overwriting
        already existing data.

        Args:
            extraParams: (list)
                A list of the extra parameters to be added to the
                config dictionary.
            extraParamValues: (list)
                A list of values of the extra parameters to be added to
                the config dictionary.

        Returns:
            (None)
                The self.config dictionary is modified.
        """
        for index, value in enumerate(extraParamValues):
            nested_keys = extraParams[index]
            current_level = self.config
            for key in nested_keys[:-1]:
                current_level = current_level.setdefault(key, {})
            current_level[nested_keys[-1]] = value

    def getConfig(self) -> dict:
        """
        A getter for the config dictionary.

        Returns:
            (dict)
                The self.config dictionary.
        """
        return self.config

    def getNetwork(self) -> pypsa.Network:
        """
        A getter for the pypsa.Network.

        Returns:
            (pypsa.Network)
                The pypsa network opened during initialization.
        """
        return self.network

    def getNetworkName(self) -> str:
        """
        A getter for the network name.

        Returns:
            (str)
                The network name.
        """
        return self.networkName
