import abc


class BackendBase(abc.ABC):
    def __init__(self, network):
        pass

    @abc.abstractstaticmethod
    def transformProblemForOptimizer(network):
        pass

    @abc.abstractstaticmethod
    def transformSolutionToNetwork(network, transformedProblem, solution):
        pass

    @abc.abstractmethod
    def optimize(self, transformedProblem):
        pass

    @abc.abstractmethod
    def getMetaInfo(self):
        pass
