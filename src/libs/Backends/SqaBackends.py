from ast import literal_eval

# try import from local .so
# Error message for image: herrd1/siquan:latest 
# /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by /energy/libs/Backends/siquan.cpython-39-x86_64-linux-gnu.so)`
import pypsa

try:
    from . import siquan
# try import from installed module siquan
except ImportError:
    import siquan


from .InputReader import InputReader
from .IsingPypsaInterface import IsingBackbone
from .BackendBase import BackendBase
import time
import random


class ClassicalBackend(BackendBase):
    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        self.solver = siquan.DTSQA()

    def transformProblemForOptimizer(self) -> None:
        """
        Initializes an IsingInterface-instance, which encodes the Ising Spin Glass 
        Problem, using the network to be optimized.

        Returns:
            (IsingBackbone) The IsingInterface-instance, which encodes the Ising Spin Glass Problem.
        """
        print("transforming problem...")
        self.transformedProblem = IsingBackbone.buildIsingProblem(
                        network=self.network,
                        config=self.config["IsingInterface"]
                        )
        return self.transformedProblem
    def transformSolutionToNetwork(self) -> pypsa.Network:
        """
        Encodes the optimal solution found during optimization and stored in self.output 
        into a pypsa.Network. 

        It reads the solution stored in the optimizer instance and prints some information
        about it. Then it writes to the netowrk

        Returns:
            (pypsa.Network) The optimized network.
        """
        self.printReport()
        # self.transformedProblem.addSQASolutionToNetwork(
        #     network, solution["state"]
        # )

        return self.network

    def optimize(self) -> None:
        """
        Optimizes the problem encoded in the IsingBackbone-Instance.

        It uses the siquan solver which parameters can be set using self.config. Configuration
        of the solver is delegated to a method that can be overwritten in child classes

        Returns:
            (None) The optimized solution is stored in the self.output dictionary.
        """
        print("starting optimization...")
        self.configureSolver()
        tic = time.perf_counter()
        result = self.solver.minimize(
            self.transformedProblem.siquanFormat(),
            self.transformedProblem.numVariables(),
        )
        self.output["results"]["optimizationTime"] = time.perf_counter() - tic
        # parse the entry in "state" before using it
        result["state"] = literal_eval(result["state"])
        self.writeResultsToOutput(result)
        print("done")

    def getHSchedule(self):
        """
        A method for getting the transverse field schedule in the configuration.

        For classical annealing, there is no transverse field, so it always return
        the same config string. Overwriting this method can be used to get non-zero
        transverse fields
        
        Returns:
            (str) the configuration string of the siquan solver according to the config dict
        """
        return "[0]"

    def configureSolver(self):
        """
        reads and sets siquan solver parameter from the config dict 

        solver configuration si read from the config dict and default values are used if it is
        not specifified in the configuration. These default values are not suitable for solving
        large problems. Since classical annealing and simulated quantum annealing only differs
        in the transverse field, setting that field is in its own method so it can be overwritten
        
        Returns:
            (None) modifies self.solver and sets hyperparameters
        """
        siquanConfig = self.config["SqaBackend"]
        self.solver.setSeed(siquanConfig.get("seed", random.randrange(10 ** 6)))
        self.solver.setHSchedule(self.getHSchedule())
        self.solver.setTSchedule(siquanConfig.get("temperatureSchedule", "[0.1,iF,0.0001]"))
        self.solver.setTrotterSlices(int(siquanConfig.get("trotterSlices", 32)))
        self.solver.setSteps(int(siquanConfig.get("optimizationCycles", 16)))

    
    def printSolverspecificReport(self):
        """
        prints additional information about the solution that is solver specific

        the only non-generic information is the energy of the solution regarding the ising 
        spin glass formulation.
        
        Returns:
            (None) no side effects other than printing
        """
        
        print(
            f"Total energy cost of QUBO (with constant terms): {self.output['results']['totalCost']}"
        )
    

    def writeResultsToOutput(self, result):
        """
        This writes solution specific values of the optimizer result and the ising spin glass
        problem solution the output dictionary. Parse the value to the key "state" via
        literal_eval before calling this function.
        
        Args:
            result: (dict) the python dictionary returned from the sqa solver

        Returns:
            (None) modifies self.output with solution specific parameters and values
        """
        for key in result:
            self.output["results"][key] = result[key]
        self.output["results"] = {
                        **self.output["results"],
                        **self.transformedProblem.generateReport(result["state"])
                    }


class SqaBackend(ClassicalBackend):
    def getHSchedule(self):
        """
        method for setting the transverse field when configuring the siquan solver
        
        Returns:
            (str) returns the configuration string of the config dict for the transverse field
        """
        return self.config["SqaBackend"].get("transverseFieldSchedule", "[8.0,0.0]")
