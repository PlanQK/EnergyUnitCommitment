import random
import time

from ast import literal_eval

from .InputReader import InputReader
from .IsingPypsaInterface import IsingBackbone
from .BackendBase import BackendBase

# try import from local .so
# Error message for image: herrd1/siquan:latest
# /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
# (required by /energy/libs/Backends/siquan.cpython-39-x86_64-linux-gnu.so)`
try:
    from . import siquan
# try import from installed module siquan
except ImportError:
    import siquan


class ClassicalBackend(BackendBase):
    """
    A class for solving the unit commitment problem using classical
    annealing. This is done using the SiQuAn solver and setting the
    transverse Field to zero.
    """

    def __init__(self, reader: InputReader):
        """
        Constructor for the ClassicalBackend class. It requires an
        InputReader, which handles the loading of the network and
        configuration file.

        Args:
            reader: (InputReader)
                 Instance of an InputReader, which handled the loading
                 of the network and configuration file.
        """
        super().__init__(reader=reader)
        self.solver = siquan.DTSQA()

    def transformProblemForOptimizer(self) -> None:
        """
        Initializes an IsingInterface-instance, which encodes the Ising
        Spin Glass Problem, using the network to be optimized.

        Returns:
            (None)
                Add the IsingInterface-instance to
                self.transformedProblem.
        """
        print("transforming problem...")
        self.transformedProblem = IsingBackbone.buildIsingProblem(
            network=self.network,
            config=self.config["IsingInterface"]
        )

    def transformSolutionToNetwork(self) -> None:
        """
        Encodes the optimal solution found during optimization and
        stored in self.output into a pypsa.Network. It reads the
        solution stored in the optimizer instance, prints some
        information regarding it to stdout and then writes it into a
        network, which is then saved in self.output.

        Returns:
            (None)
                Modifies self.output with the outputNetwork.
        """
        self.printReport()

        outputNetwork = self.transformedProblem.setOutputNetwork(
            solution=self.output["results"]["state"])
        outputDataset = outputNetwork.export_to_netcdf()
        self.output["network"] = outputDataset.to_dict()

    def optimize(self) -> None:
        """
        Optimizes the problem encoded in the IsingBackbone-Instance.
        It uses the siquan solver which parameters can be set using
        self.config. Configuration of the solver is delegated to a
        method that can be overwritten in child classes.

        Returns:
            (None)
                The optimized solution is stored in the self.output
                dictionary.
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

    def getHSchedule(self) -> str:
        """
        A method for getting the transverse field schedule in the
        configuration. For classical annealing, there is no transverse
        field, so it always returns the same config string. Overwriting
        this method can be used to get non-zero transverse field.
        
        Returns:
            (str)
                The configuration string for the transverse field of the
                siquan solver for classical annealing.
        """
        return "[0]"

    def configureSolver(self) -> None:
        """
        Reads and sets siquan solver parameter from the config dict.
        Solver configuration is read from the config dict and default
        values are used if it is not specifified in the configuration.
        These default values are not suitable for solving large
        problems. Since classical annealing and simulated quantum
        annealing only differs in the transverse field, setting that
        field is in its own method so it can be overwritten.
        
        Returns:
            (None)
                Modifies self.solver and sets hyperparameters
        """
        siquanConfig = self.config["SqaBackend"]
        self.solver.setSeed(siquanConfig.get("seed",
                                             random.randrange(10 ** 6)))
        self.solver.setHSchedule(self.getHSchedule())
        self.solver.setTSchedule(siquanConfig.get("temperatureSchedule",
                                                  "[0.1,iF,0.0001]"))
        self.solver.setTrotterSlices(int(siquanConfig.get("trotterSlices",
                                                          32)))
        self.solver.setSteps(int(siquanConfig.get("optimizationCycles", 16)))

    def print_solver_specific_report(self) -> None:
        """
        Prints additional information about the solution that is solver
        specific.
        The only non-generic information is the energy of the solution
        regarding the Ising spin glass formulation.
        
        Returns:
            (None)
                Prints information to stdout.
        """
        print(
            f"Total energy cost of QUBO (with constant terms): "
            f"{self.output['results']['totalCost']}"
        )
        print("---")
        print(f"Sqa runtime: {self.output['results']['runtime_sec']}")
        print(f"Sqa runtime cycles: {self.output['results']['runtime_cycles']}")
        print(f"Ising Interactions: {len(self.transformedProblem.problem)}")

    def writeResultsToOutput(self, result: dict) -> None:
        """
        This writes solution specific values of the optimizer result
        and the Ising spin glass problem solution the self.output.
        Parse the value to the key "state" via literal_eval before
        calling this function.
        
        Args:
            result: (dict)
                The python dictionary returned from the sqa solver.

        Returns:
            (None)
                Modifies self.output with solution specific parameters
                and values.
        """
        for key in result:
            self.output["results"][key] = result[key]
        self.output["results"] = {
            **self.output["results"],
            **self.transformedProblem.generateReport(
                result["state"]
            )
        }


class SqaBackend(ClassicalBackend):
    """
    A class for solving the unit commitment problem using classical
    annealing. This is done using the SiQuAn solver and setting the
    transverse Field to a value given in the configuration file. It
    inherits from ClassicalBackend and only overwrites the method
    getHSchedule.
    """

    def getHSchedule(self) -> str:
        """
        A method for getting the transverse field schedule from the
        configuration and returns it.

        Returns:
            (str)
                The configuration of the 'transverseFieldSchedule', to
                set in siquan solver, according to self.config.
        """
        return self.config["SqaBackend"].get("transverseFieldSchedule",
                                             "[8.0,0.0]")
