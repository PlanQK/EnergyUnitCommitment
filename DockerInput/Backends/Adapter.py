import json
import yaml


class Adapter:
    """
    This class is an adapter to obtain the configuration dictionary dependent on the input format
    """
    def __init__(self, data):
        self.config = {}
        self.setConfig(data)

    def setConfig(self, *args):
        pass

    def getConfig(self) -> dict:
        return self.config

    @classmethod
    def makeAdapter(self, data):
        if isinstance(data, dict):
            return DictAdapter(data)
        elif isinstance(data, str):
            AdapterFormats = {
                    "yaml" : YamlAdapter,
                    "json" : JsonAdapter
            }
            return AdapterFormats[data[-4:]](data)
        raise ValueError("input can't be read")


class DictAdapter(Adapter):
    def setConfig(self, config: dict):
        self.config = config


class JsonAdapter(Adapter):
    def setConfig(self, path: str):
        with open(path) as file:
            self.config = json.load(file)


class YamlAdapter(Adapter):
    def setConfig(self, path: str):
        with open(path) as file:
            self.config = yaml.safe_load(file)


class StandardAdapter(Adapter):
    def setConfig(self):
        self.config = {}


class EnvAdapter(Adapter):
    def setConfig(self):
        pass
