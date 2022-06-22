"""Basic definitions of the response class that wrap the serialization of optimization results.
These responses can be used to provide an HTTP response in the case of the service or save 
the information to the disk
"""

import json
from abc import abstractmethod
from datetime import datetime


class Response:
    def to_json(self):
        """
        Converts the data of the response into json format and returns it as a string
    
        Returns:
            (str) The serialized response as a string
        """
        return json.dumps(self, default=lambda o: getattr(o, '__dict__', str(o)), sort_keys=True)

    @abstractmethod
    def save_to_json_local_docker(self, folder: str = "Problemset/"):
        pass


class ErrorResponse(Response):
    """
    Represents an error to be passed back to the caller.

    Args:
        code (int): HTTP status code to be passed back to the caller
        detail (str): Error message to be passed back to the caller
    """

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail

    def save_to_json_local_docker(self, folder: str = "Problemset/"):
        error = {"status_code": self.code,
                 "message": self.detail}
        now = datetime.today()
        date_time_str = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
        with open(f"{folder}error_code_{self.code}_{date_time_str}.json", "w") as write_file:
            json.dump(error, write_file, indent=2, default=str)


class ResultResponse(Response):
    """
    Represents the result to be passed back to the caller.

    Args:
        result (dict): Json serializable Dict to be passed back to the caller
    """

    def __init__(self, result: dict, metadata: dict = None):
        self.result = result
        self.metadata = metadata
        self.fileName = result["file_name"]

    def save_to_json_local_docker(self, folder: str = "Problemset/"):
        with open(f"{folder}{self.fileName}", "w") as write_file:
            json.dump(self.result, write_file, indent=2, default=str)
