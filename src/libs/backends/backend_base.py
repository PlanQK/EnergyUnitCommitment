"""This module contains the BackendBase class, which describes the interface
a solver of the unit commitment problem has to satisfy. It also sets up
various attributes that are all solvers use"""

import abc

from .input_reader import InputReader
from datetime import datetime


class BackendBase(abc.ABC):
    """
    The BackendBase class is an interface that the solvers of the unit
    commitment problem have to adhere it. It also initializes some 
    attributes that are the same for all solvers.

    How an object that inherits from this class is used to formulate and
    optimize the unit commitment problem can be seen in `program.py`
    """
    def __init__(self, reader: InputReader):
        self.output = None
        self.reader = reader
        self.network = reader.get_network()
        self.network_name = reader.get_network_name()
        self.config = reader.get_config()
        self.config_name = reader.get_config_name()
        self.file_name = reader.get_file_name()
        self.setup_output_dict()
        self.transformed_problem = None

    @abc.abstractmethod
    def transform_problem_for_optimizer(self) -> None:
        """
        This transforms the problem given as a pypsa network
        into the corresponding data structure the algorithm this
        class implements 
    
        Returns:
            (None) 
                This sets the attribute `self.transformed_problem`
                to the correct value
        """
        pass

    @abc.abstractmethod
    def optimize(self) -> None:
        """
        This method calls the optimization method of the solver after
        the unit commitment problem has been transformed. The result is
        written to the attribute `self.result`

        Returns:
            (None) 
                Modifies `self.output` and can have side effects on other
                attributes.
        """
        pass

    def process_solution(self) -> None:
        """
        A hook for postprocessing of the solution if this is required

        Returns:
            (type) 
                Returns nothing, but can have side effects if this method
                is overwritten is a subclass.
        """
        self.output["results"]["postprocessing_time"] = 0.0

    def transform_solution_to_network(self):
        """
        This writes the solution of the optimization run into a copy of
        the original pypsa network.
        """
        pass

    @classmethod
    def create_optimizer(cls, reader: InputReader):
        """
        A class method to instantiate the correct subclass of the 
        optimizer based on the configuration in the reader. 

        This method returns an instance of the calling class. If the optimizer
        are organized as subclasses of an abstract class, this method has
        to be overwritten to return a class that can be instantiated.

        Args:
            reader: (InputReader)
                An instance of the InputReader. Any input that the optimizer
                accepts has to be passed by this instance.
        """
        return cls(reader)

    def check_input_size(self, limit: int):
        """
        Check if the size of the problem is currently allowed or if
        an estimation of the runtime exceeds some arbitrary limit
        stop the run.

        Args:
            limit: (int)
                A parameter for the upper limit that an optimization run is
                allowed to take. This is not meant to be how many seconds
                it can take.

        Returns:
            (None) 
                If the input takes to long to optimize, this raises
                an error.
        """
        pass

    def get_config(self) -> dict:
        """
        A getter for the config-dictionary.

        Returns:
            (dict) The config used for the current problem.
        """
        return self.config

    @staticmethod
    def get_time() -> str:
        """
        A getter for the current time.

        Returns:
            (str) The current time in the format YYYY-MM-DD_hh-mm-ss
        """
        now = datetime.today()
        return f"{now.year}-{now.month}-{now.day}" \
               f"_{now.hour}-{now.minute}-{now.second}"

    def get_output(self) -> dict:
        """
        A getter for the output-dictionary. Before returning the
        dictionary the end time is added to it.

        Returns:
            (dict) The output (result) of the current problem.
        """
        self.output["end_time"] = self.get_time()
        return self.output

    def print_solver_specific_report(self):
        """
        A subclass may overwrite this method in order to print
        additional information after the generic information about
        the optimization run.
    
        Returns:
            (None) prints additional output when `print_report` is called
        """
        pass

    def print_report(self) -> None:
        """
        Prints a short report with general information about the
        solution.

        Returns:
            (None)
        """
        print(f"\n--- General information of the solution ---")
        print(f'Kirchhoff cost at each bus: '
              f'{self.output["results"].get("individual_kirchhoff_cost","N/A")}')
        print(f'Kirchhoff cost at each time step: '
              f'{self.output["results"].get("kirchhoff_cost_by_time","N/A")}')
        print(f'Total Kirchhoff cost: '
              f'{self.output["results"].get("kirchhoff_cost","N/A")}')
        print(f'Total power imbalance: '
              f'{self.output["results"].get("power_imbalance","N/A")}')
        print(f'Total Power generated: '
              f'{self.output["results"].get("total_power","N/A")}')
        print(f'Total marginal cost: '
              f'{self.output["results"].get("marginal_cost","N/A")}')
        self.print_solver_specific_report()
        print("---")

    def setup_output_dict(self) -> None:
        """
        Creates an 'output' attribute in self in which to save results
        and configuration data. The config entry is another dictionary
        with 3 keys: 'backend' has config data that all backends share,
        'ising_interface' has config data of the class used to convert a
        unit commitment problem into an ising spin problem and a key
        named in `backend_config` for backend specific configurations.

        Returns:
            (None) Creates the attribute `output` with a dictionary
                   containing configuration data and empty fields to
                   insert results into later on.
        """
        start_time = self.get_time()
        for backend, solver_list in self.reader.backend_to_solver.items():
            if self.config["backend"] in solver_list:
                file_name = "_".join([self.config_name,
                                      start_time + ".json"])
                if self.network_name:
                    file_name = "_".join([self.network_name,
                                          file_name])
                if self.file_name:
                    file_name = self.file_name
                self.output = {
                    "start_time": start_time,
                    "end_time": "",
                    "file_name": file_name,
                    "config": {
                        "backend": self.config["backend"],
                        "backend_type": backend,
                        "backend_config": self.config.get(backend, {}),
                        "ising_interface": self.config.get("ising_interface", {}),
                    },
                    "results": {},
                }
                return
        raise ValueError("The specified backend didn't resolve to a valid backend")
