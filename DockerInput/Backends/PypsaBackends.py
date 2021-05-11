from .BackendBase import BackendBase


class PypsaBackend(BackendBase):
    def transformProblemForOptimizer():
        pass

    def transformSolutionToNetwork():
        pass

    def optimize(self, transformedProblem):
        pass

    def getMetaInfo(self):
        pass


class PypsaFico(PypsaBackend):
    pass


class PypsaGlpk(PypsaBackend):
    pass
