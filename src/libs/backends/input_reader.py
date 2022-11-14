"""This module is for managing input and output. It provives classes for
accepting various forms of input and extract all information relevant to
the optimization run. Right now it supports the following types of input
# input
- reading a network and config file from disk, an additional input string
when calling the script that initiates the optimization
- reading jsons, one containing a serialized network, the other a config file

# output
- so far, output is saved in the backends using a dictionary, or by creating
a response object
"""

import copy
import json
import pypsa
import xarray
import yaml

from os import environ
from werkzeug.utils import secure_filename

from typing import Union

from .. import backends


class InputReader:
    """
    This class is a reader to obtain the configuration dictionary and
    pypsa.Network, dependent on the input format.
    """

    # dictionary which solver has which backend specific extra data where.
    # Keys are broader categories of backends and values are a list of solvers
    # that use that key to store additional config info.
    backend_to_solver = {
        "dwave_backend": [
            "dwave-tabu",
            "dwave-greedy",
            "dwave-hybrid",
            "dwave-qpu",
            "dwave-read-qpu",
        ],
        "pypsa_backend": ["pypsa-glpk", "pypsa-fico"],
        "sqa_backend": ["sqa", "classical", "iterative-sqa"],
        "qaoa_backend": ["qaoa"],
    }
    loaders = {
        "json": json.load,
        "yaml": yaml.safe_load
    }

    def __init__(self,
                 network: Union[str, dict, pypsa.Network],
                 config: Union[str, dict],
                 params_dict: dict = None,
                 ):
        """
        Obtain the configuration dictionary and pypsa.Network,
        dependent on the input format of network and config. The
        obtained configuration dictionary is then amended with the
        extra_params, if provided.

        Args:
            network: (Union[str, dict, pypsa.Network])
                A string, dictionary or pypsa.Network, representing the
                pypsa.Network to be used.
            config: (Union[str, dict])
                A string or dictionary to be used to obtain the
                configuration for the problem instances.
        """
        self.network, self.network_name = self.make_network(network)
        self.config = self.make_config(config)
        self.add_params_dict(params_dict)
        self.copy_to_backend_config()
        self.network.snapshots = self.network.snapshots[:self.config.get("snapshots",
                                                                         len(self.network.snapshots))]
        # print final config, but hide api tokens
        config_without_token = copy.deepcopy(self.config)
        for provider, token in self.config.get("API_token", {}).items():
            if token != "":
                config_without_token["API_token"][provider] = "*****"
        print(f"running with the following configuration {config_without_token}")

    def copy_to_backend_config(self) -> None:
        """
        Copies the backend specific configuration to the backend
        agnostic key "backend_config" where it is merged with any already
        present dictionary entries.

        Returns:
            (None)
                Modifies self.config.
        """
        self.config["backend"] = self.config.get("backend", "sqa").replace('_', '-')
        for backend_type, solver_list in self.backend_to_solver.items():
            if self.config["backend"] in solver_list:
                # expansion has guards for passing None objects as config dicts
                self.config["backend_config"] = {
                    **(self.config.get("backend_config", {}) or {}),
                    **(self.config.get(backend_type, {}) or {}),
                }
                return

    def make_network(self,
                     network: Union[str, dict, pypsa.Network]
                     ) -> [pypsa.Network, str]:
        """
        Opens a pypsa.Network using the provided network argument.

        If a string is given it is interpreted as the network name. It has to
        be stored in the folder "problemset". If a dictionary is given,
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
        # load from disk
        if isinstance(network, str):
            if environ.get("RUNNING_IN_DOCKER", False):
                network_path = "problemset/"
            else:
                network_path = "../input/networks/"
            return pypsa.Network(network_path + network), network

        # load from serialized network
        elif isinstance(network, dict):
            loaded_dataset = xarray.Dataset.from_dict(network)
            loaded_net = pypsa.Network()
            pypsa.Network.import_from_netcdf(network=loaded_net,
                                             path=loaded_dataset)
            return loaded_net, network["attrs"].get("network_name", "network_from_dict")

        # the network has been build in a previous step
        elif isinstance(network, pypsa.Network):
            return network, "no_name_network"

        raise TypeError("The network has to be given as a dictionary, "
                        "representing the netCDF format of a pypsa.Network, "
                        "an actual pypsa.Network, or a string with the name "
                        "of the pypsa.Network, which denotes the folder it is "
                        "stored in.")

    def make_config(self, input_config: Union[str, dict]) -> dict:
        """
        Converts an input_config file into a dictionary.

        If the input is a dictionary it will be passed as the output.
        If it is a string, the filetype will be determined, the file opened and
        stored in a dictionary.
        If the environment variable "RUNNING_IN_DOCKER",
        is set, it is assumed that program runs in the container made by the
        dockerfile in this repo and the folder containing the file has been mounted
        to `/energy/configs`. Otherwise, the files in `input/configs` are searched.
        The supported filetypes are json and yaml.

        Args:
            input_config: (Union[str, dict])
                A string or dictionary to be used to obtain the
                configuration for the problem instances.

        Returns:
            (dict)
                The config stored in a dictionary.
        """
        self.config_name = ""
        if isinstance(input_config, dict):
            result = input_config
        else:
            # input_config is assumed to be the path if it is not a dict
            input_config = secure_filename(input_config)
            filename, filetype = input_config.split(".")[-2:]
            self.config_name = filename
            try:
                loader = self.loaders[filetype]
            except KeyError:
                raise KeyError(f"The file format {filetype} doesn't match any supported "
                               f"format. The supported formats are {list(self.loaders.keys())}")
            if environ.get("RUNNING_IN_DOCKER", False):
                config_path = "configs/"
            else:
                config_path = "../input/configs/"
            with open(config_path + input_config) as file:
                result = loader(file)
        base_dict = {
            "API_token": {},
            "backend": "sqa",
            "ising_interface": {},
            "backend_config": {},
        }
        return {**base_dict, **result}

    def add_params_dict(self, params_dict: dict = None, current_level: dict = None) -> None:
        """
        Writes extra parameters into the config dictionary, overwriting
        already existing data.

        Args:
            params_dict: (dict)
                a dictionary containing parameters to be put into the
                config, overwriting existing values
            current_level: (dict)
                The current dictionary in the nested `self.config` that is passed
                to write the values in `params_dict` into it recursively
                with the correct nested keywords

        Returns:
            (None)
                The self.config dictionary is modified.
        """
        if params_dict is None:
            return
        if current_level is None:
            current_level = self.config
        for config_key, config_value in params_dict.items():
            if isinstance(config_value, dict):
                self.add_params_dict(config_value, current_level.setdefault(config_key, {}))
            else:
                current_level[config_key] = config_value

    def get_optimizer_class(self) -> 'backends.BackendBase':
        """
        Returns the corresponding optimizer class that is specified
        in the `config` attribute of this instance

        Returns:
            (BackendBase)
                The optimizer class that corresponds to the config value
        """
        gan_backends = {
            "classical": backends.ClassicalBackend,
            "sqa": backends.SqaBackend,
            "iterative-sqa": backends.SqaIterator,
            "dwave-tabu": backends.DwaveTabu,
            "dwave-greedy": backends.DwaveSteepestDescent,
            "pypsa-glpk": backends.PypsaGlpk,
            "pypsa-fico": backends.PypsaFico,
            "dwave-hybrid": backends.DwaveCloudHybrid,
            "dwave-qpu": backends.DwaveCloudDirectQPU,
            "dwave-read-qpu": backends.DwaveReadQPU,
            "qaoa": backends.QaoaQiskit,
        }
        return gan_backends[self.config["backend"]]

    def get_config(self) -> dict:
        """
        A getter for the config dictionary.

        Returns:
            (dict)
                The self.config dictionary.
        """
        return self.config

    def get_config_name(self) -> str:
        """
        A getter for the config dictionary name.

        Returns:
            (dict)
                The self.config dictionary.
        """
        return self.config_name

    def get_file_name(self) -> str:
        """
        A getter for the file the results gets dumped too

        Returns:
            (str)
                the file name without path where the results are dumped
        """
        return self.config.get("file_name", "")

    def get_network(self) -> pypsa.Network:
        """
        A getter for the pypsa.Network.

        Returns:
            (pypsa.Network)
                The pypsa network opened during initialization.
        """
        return self.network

    def get_network_name(self) -> str:
        """
        A getter for the network name.

        Returns:
            (str)
                The network name.
        """
        return self.network_name
