import json
import yaml


class Adapter:
    """
    This class is an adapter to obtain the configuration dictionary dependent on the input format
    """
    def __init__(self):
        self.config = {}
        self.setConfig()

    def setConfig(self, *args):
        pass

    def getConfig(self) -> dict:
        return self.config


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
