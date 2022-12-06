"""Basic definitions of the response class that wrap the serialization of optimization results.
These responses can be used to provide an HTTP response in the case of the service or save
the information to the disk
"""

import json
from abc import abstractmethod
from datetime import datetime


class Response:
    """
    A class for wrapping the serialization of optimization results.
    """

    def to_json(self) -> str:
        """
        Converts the data of the response into json format and returns it as a string

        Returns:
            (str) The serialized response as a string
        """
        return json.dumps(
            self,
            default=lambda o: getattr(o, "__dict__", str(o)),
            sort_keys=True,
            indent=2,
        )

    @abstractmethod
    def dump_results(self, folder: str = "problemset/") -> None:
        """
        An abstract method to be overwritten in child classes. The behaviour of the method
        is to write the contents of the response to the disk. This is needed for saving the
        results of an optimization if you ran it locally in a docker container

        Args:
            folder: (str) location where the information of the response is dumped. The default
                option is the mount point that the makefile uses for mounting the network folder
        Returns:
            (None) Writes a file to the disk.
        """
        pass


class ErrorResponse(Response):
    """
    Represents an error thrown during the optimization to be passed back to the caller.

    Args:
        code (str): HTTP status code to be passed back to the caller
        detail (str): Error message to be passed back to the caller
    """

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail

    def dump_results(self, folder: str = "problemset/") -> None:
        """
        Saves the thrown error to the disk. Instead of using the name specified
        in the config for the result, this will instead use a name using the
        time when the error was encountered and which error code was thrown.

        Args:
            folder: (str) the location where to save the response. The default
                path is the mount point of the makefile for the networks

        Returns:
            (None) Saves the error code to the mount point of the networks
        """
        error = {"status_code": self.code, "message": self.detail}
        now = datetime.today()
        date_time_str = (
            f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
        )
        with open(
            f"{folder}error_code_{self.code}_{date_time_str}.json",
            "w",
            encoding="utf-8",
        ) as write_file:
            json.dump(error, write_file, indent=2, default=str)


class ResultResponse(Response):
    """
    Represents the result of a successful optimization to be passed back to the caller.

    Args:
        result (dict): serializable dictionary to be passed back to the caller
        metadata (dict): serializable dictionary containing meta data
    """

    def __init__(self, result: dict, metadata: dict = None):
        self.result = result
        self.metadata = metadata

    def dump_results(self, folder: str = "problemset/") -> None:
        """
        Saves the result of the local optimization in a docker container.

        Args:
            folder: (str)
                The location where to save the response. The default
                path is the mount point of the makefile for the networks

        Returns:
            (None) Saves the optimization result to the mount point of the networks
        """
        with open(
            f"{folder}{self.result['file_name']}", "w", encoding="utf-8"
        ) as write_file:
            json.dump(self.result, write_file, indent=2, default=str)
