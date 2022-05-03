import abc

from .InputReader import InputReader
from datetime import datetime


class BackendBase(abc.ABC):
    def __init__(self, reader: InputReader):
        self.reader = reader
        self.network = reader.getNetwork()
        self.networkName = reader.getNetworkName()
        self.config = reader.getConfig()
        self.setupOutputDict()

    @abc.abstractmethod
    def transformProblemForOptimizer(self, network):
        pass

    @abc.abstractstaticmethod
    def transformSolutionToNetwork(network, transformedProblem, solution):
        pass

    @abc.abstractmethod
    def processSolution(self, network, transformedProblem, solution):
        pass

    @abc.abstractmethod
    def optimize(self, transformedProblem):
        pass

    @abc.abstractmethod
    def validateInput(self, path, network):
        pass

    @abc.abstractmethod
    def handleOptimizationStop(self, path, network):
        pass

    def getConfig(self) -> dict:
        return self.config

    def getTime(self) -> str:
        now = datetime.today()
        return f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"

    def getOutput(self) -> dict:
        self.output["end_time"] = self.getTime()
        return self.output

    def printSolverspecificReport(self):
        pass

    def printReport(self):
        print(f'\n--- General information of the solution ---')
        print(f'Kirchhoff cost at each bus: {self.output["results"].get("individualKirchhoffCost","N/A")}')
        print(f'Total Kirchhoff cost: {self.output["results"].get("kirchhoffCost","N/A")}')
        print(f'Total power imbalance: {self.output["results"].get("powerImbalance","N/A")}')
        print(f'Total Power generated: {self.output["results"].get("totalPower","N/A")}')
        print(f'Total marginal cost: {self.output["results"].get("marginalCost","N/A")}')
        self.printSolverspecificReport()
        print('---')


    def setupOutputDict(self):
        """
        creates an 'output' attribute in self in which to save results and configuration
        data. The config entry is another dictionary with 3 keys: 'Backend' has config
        data that all backends share, 'IsingInterface' has config data of the class
        used to convert a unit commitment problem into an ising spin problem
        and a key named in `BackendToSolver` for backend specific configurations
        
        Returns:
            (None) (over)writes the attribute `output` with a dicitionary containing
            configuration data and empty fields to insert results into later on.
        """
        startTime = self.getTime()
        # dictionary which solver has which backend specific extra data where. Keys are
        # broader categories of backends and values are a list of solvers that use that
        # key to store additional config info
        # TODO flatten structure, by making new key, backend category and a generic key
        ## for additional configs for this category of solvers
        BackendToSolver = {
                "DWaveBackend": ["dwave-tabu", "dwave-greedy", "dwave-hybrid", "dwave-qpu", "dwave-read-qpu"],
                "PypsaBackend": ["pypsa-glpk", "pypsa-fico"],
                "SqaBackend": ["sqa", "classical"],
                "QaoaBackend": ["qaoa"],
        }
        for backend, solverList in BackendToSolver.items():
            if self.config["Backend"] in solverList:
                self.output = {"start_time": startTime,
                               "end_time": None,
                               "file_name":  "_".join([
                                        self.networkName,
                                        self.config["Backend"],
                                        startTime + ".json"
                                        ]),
                               "config": {
                                        "Backend": self.config["Backend"],
                                        "BackendType": backend,
                                        "BackendConfig": self.config[backend],
                                        "IsingInterface": self.config["IsingInterface"],
                                         
                                      },
                               "components": {},
                               "network": {},
                               "results": {},
                               }
                return
                    
