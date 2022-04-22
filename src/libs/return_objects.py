import json


class Response:
    def to_json(self):
        return json.dumps(self, default=lambda o: getattr(o, '__dict__', str(o)), sort_keys=True)


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


class ResultResponse(Response):
    """
    Represents the result to be passed back to the caller.

    Args:
        result (dict): Json serializable Dict to be passed back to the caller
    """

    def __init__(self, result: dict, metadata: dict = None):
        self.result = result
        self.metadata = metadata
