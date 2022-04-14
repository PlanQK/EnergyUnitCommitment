import json
import yaml
import pypsa


class Adapter:
    """
    This class is an adapter to obtain the configuration dictionary dependent on the input format
    """
    def __init__(self, data, params):
        self.network = self.makeNetwork(data)
        self.config = self.makeConfig(params)

    def getConfig(self) -> dict:
        return self.config

    def getNetwork(self):
        return self.network


    @classmethod
    def makeAdapter(self, **kwargs):
        params = kwargs["params"]
        data = kwargs["data"]
        return Adapter(data=data, params=params)
    

    def makeNetwork(self, data):
        if isinstance(data, str):
            return pypsa.Network(f"Problemset/" + data)
        raise NotImplementedError
                
    
    def makeConfig(self, params):
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

