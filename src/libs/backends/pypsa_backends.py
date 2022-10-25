"""This module uses pypsa's MILP formulation to generate the problem
and then solves it using one of the installed optimizers. The possible
choices for optimizers are
- GLPK
- FicoXpress
"""

import pypsa

from .backend_base import BackendBase
from .input_reader import InputReader


class PypsaBackend(BackendBase):
    """
    A class for solving the unit commitment problem using classical
    optimization algorithms. This is done using the solver shipped with the
    Pypsa package.
    """

    def __init__(self, reader: InputReader):
        """
        Constructor for a PypsaBackend. It requires an InputReader,
        which handles the loading of the network and configuration file.

        Args:
            reader: (InputReader)
                 Instance of an InputReader, which handled the loading
                 of the network and configuration file.
        """
        super().__init__(reader=reader)
        if self.config["backend_config"].get("timeout", -1) < 0:
            self.config["backend_config"]["timeout"] = 3600

    def transform_problem_for_optimizer(self) -> None:
        """
        Sets up the linear programming optimizer and the linear
        programming formulation.
        The optimizer can be accessed at `self.opt` and then linear
        program can be accessed at `self.transformed_problem`.
        
        Returns:
            (None)
                Modifies `self.opt` and `self.transformed_problem`.
        """
        print("transforming problem...")
        self.network.generators.committable = True
        self.network.generators.p_nom_extendable = False

        # avoid committing a generator and setting output to 0 
        self.network.generators_t.p_min_pu = self.network.generators_t.p_max_pu
        self.network.generators.p_min_pu = self.network.generators.p_max_pu
        self.transformed_problem = pypsa.opf.network_lopf_build_model(
            network=self.network,
            snapshots=self.network.snapshots,
            formulation="kirchhoff"
        )
        self.opt = pypsa.opf.network_lopf_prepare_solver(
            network=self.network,
            solver_name=self.config["backend_config"].get("solver_name", "glpk"))
        self.opt.options["tmlim"] = self.config["backend_config"]["timeout"]

    def check_input_size(self, limit: float = 60.0):
        """
        sets the maximum run time to the limit given as an argument

        Args:
            limit: the upper limit on run time. In this case, it doesn't stop
            the optimization, but sets an option in the solver to stop 
            the optimization after the limit is exceeded

        Returns:
            (None)
                Sets the timelimit of the MILP solver
        """
        self.opt.options["tmlim"] = min(limit,
                                        self.config["backend_config"]["timeout"])

    def transform_solution_to_network(self) -> pypsa.Network:
        """
        (Not implemented yet) A method to write an optimization result
        into a pypsa network.
        If a solution is found, this also prints information about the
        solution.

        Returns:
            (pypsa.Network)
                Returns a pypsa network which solves the unit commitment
                problem.
        """
        print("Writing from pyomo model to network is not implemented")
        return self.network

    def print_report(self):
        """
        Prints a short report with general information if the problem is feasible
        and a message that it is infeasible if it is not feasible
    
        Returns:
            (None) 
        """
        if self.output["results"]["termination_condition"] == "infeasible":
            print("no feasible solution was found, stop writing to network")
        else:
            super().print_report()

    def write_result_to_output(self, solverstring: str) -> None:
        """
        Write the solution and information about it into the `self.output`
        dictionary.
    
        Args:
            solverstring: (str)
                The string information (in string format) returned by
                the solver of the linear program.
        Returns:
            (None)
                Modifies `self.output["results"]`.
        """
        self.output["results"]["optimizationTime"] \
            = solverstring.splitlines()[-1].split()[1]
        self.output["results"]["termination_condition"] \
            = solverstring.splitlines()[-7].split()[2]
        if self.output["results"]["termination_condition"] != "infeasible":
            total_cost = 0
            total_power = 0
            for key, val in self.transformed_problem.generator_p.get_values(
            ).items():
                total_cost += self.network.generators["marginal_cost"].loc[
                                 key[0]] * val
                total_power += val
            self.output["results"]["marginal_cost"] = total_cost
            self.output["results"]["total_power"] = total_power

            self.output["results"]["unit_commitment"] = {
                str(gen): value for gen, value in
                self.transformed_problem.generator_status.get_values().items()
            }
            self.output["results"]["powerflow"] = {
                str(line[1:]): value for line, value in
                self.transformed_problem.passive_branch_p.get_values().items()
            }
            # solver only allows feasible solutions 
            self.output["results"]["kirchhoff_cost"] = 0
            self.output["results"]["power_imbalance"] = 0

    def optimize(self) -> None:
        """
        Solves the linear program stored in self using the solver stored
        in self.
        
        Returns:
            (None)
                Writes the solution into self.output["results"].
        """
        print("starting optimization...")
        solution = self.opt.solve(self.transformed_problem)
        solution.write()
        solverstring = str(solution["solver"])
        self.write_result_to_output(solverstring=solverstring)
        print("done")


class PypsaFico(PypsaBackend):
    pass


class PypsaGlpk(PypsaBackend):
    pass
