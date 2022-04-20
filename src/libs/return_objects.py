#!/usr/bin/env python3
# -*- coding: utf8 -*-
import json
import os
import sys
from abc import abstractmethod
from typing import Dict


class Response:
    @abstractmethod
    def send(self):
        pass


class ErrorResponse(Response):
    """
    Represents an error to be passed back to the caller.

    Args:
        status_code (int): HTTP status code to be passed back to the caller
        error_message (str): Error message to be passed back to the caller
    """

    def __init__(self, status_code: int, error_message: str):
        self.status_code = status_code
        self.error_message = error_message

    def send(self):
        sys.stderr.write(str({"status_code": self.status_code, "message": self.error_message}))
        sys.stderr.flush()
        if os.getenv("DEBUG") != "true":
            os._exit(1)


class ResultResponse(Response):
    """
    Represents the result to be passed back to the caller.

    Args:
        result (Dict): Json serializable Dict to be passed back to the caller
    """

    def __init__(self, result: Dict):
        self.result = result

    def send(self):
        sys.stdout.write(str(self.result))
        sys.stdout.flush()
        if os.getenv("DEBUG") != "true":
            os._exit(0)


class ResultFileResponse(Response):
    def __init__(self, result: Dict):
        self.result = result
        self.fileName = result["file_name"]

    def send(self):
        with open(f"Problemset/{self.fileName}", "w") as write_file:
            json.dump(self.result, write_file, indent=2, default=str)
        if os.getenv("DEBUG") != "true":
            os._exit(0)