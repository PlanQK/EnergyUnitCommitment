import time

from .BackendBase import BackendBase
import pypsa

from .InputReader import InputReader


class PypsaBackend(BackendBase):
    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        if self.config["BackendConfig"].get("timeout", -1) < 0:
            self.config["BackendConfig"]["timeout"] = 3600

    def transformProblemForOptimizer(self):
        """
        sets up the linear programming optimizer and the linear programming
        formulation.

        The optimizer can be accessed at `self.opt` and then linear program
        can be accessed at `self.transformedProblem` 
        
        Returns:
            (None) Modifies `self.opt` and `self.transformedProblem`
        """
        print("transforming problem...") 
        # self.network = network.copy() TODO: maybe deepcopy before setting things in network.
        self.network.generators.committable = True
        self.network.generators.p_nom_extendable = False

        # avoid committing a generator and setting output to 0 
        self.network.generators_t.p_min_pu = self.network.generators_t.p_max_pu
        self.transformedProblem = pypsa.opf.network_lopf_build_model(
                network=self.network,
                snapshots=self.network.snapshots,
                formulation="kirchhoff"
        )
        self.opt = pypsa.opf.network_lopf_prepare_solver(
                                    self.network,
                                    solver_name=self.config["BackendConfig"]["solver_name"]
                                    )
        self.opt.options["tmlim"] = self.config["BackendConfig"]["timeout"]

    def transformSolutionToNetwork(self):
        """
        A method to write an optimization result into a pypsa network
    
        If a solution is found, this also prints information about the solution. 

        Returns:
            (pypsa.Network) returns a pypsa network which solves the unit commitment problem
        """
        # TODO implement write from pyomo
        print("Writing from pyoyo model to network is not implemented")

        if self.output["results"]["terminationCondition"] == "infeasible":
            print("no feasible solution was found, stop writing to network")
        else:
            self.printReport()
        return self.network

    def writeResultToOutput(self, solverstring):
        """
        Write solution and info about it into the the `output["results"]` dictionary
    
        Args:
            solverstring: (str) the string that the used solver of the linear program uses returns
        Returns:
            (None) modifies `output["results"]`
        """
        self.output["results"]["optimizationTime"] = solverstring.splitlines()[-1].split()[1]
        self.output["results"]["terminationCondition"] = solverstring.splitlines()[-7].split()[2]
        if self.output["results"]["terminationCondition"] != "infeasible":
            # TODO get result better out of model?
            totalCost = 0
            totalPower = 0
            for key, val in  self.transformedProblem.generator_p.get_values().items():
                totalCost += self.network.generators["marginal_cost"].loc[key[0]] * val
                totalPower += self.network.generators.p_nom.loc[key[0]] * val
            self.output["results"]["marginalCost"] = totalCost
            self.output["results"]["totalPower"] = totalPower
        
            self.output["results"]["unitCommitment"] = {
                    gen[0] : value for gen,value in self.transformedProblem.generator_status.get_values().items()
            }
            # list of indices of active generators
            self.output["results"]["state"] = [
                    idx for idx in range(len(self.network.generators))
                    if self.output["results"]["unitCommitment"][self.network.generators.index[idx]] == 1.0
            ]
            self.output["results"]["powerflow"] = {
                    line[1] : value 
                    for line,value in self.transformedProblem.passive_branch_p.get_values().items()
            }
            # solver only allows feasible solutions 
            self.output["results"]["kirchhoffCost"] = 0
            self.output["results"]["powerImbalance"] = 0

    def optimize(self):
        """
        solves the linear program stored in self using the solver stored in self
        
        Returns:
            (None) Writes the solution into self.output["results"]
        """
        print("starting optimization...")
        solution = self.opt.solve(self.transformedProblem)
        solution.write()
        solverstring = str(solution["Solver"])
        self.writeResultToOutput(solverstring=solverstring)
        print("done")


class PypsaFico(PypsaBackend):
    pass

class PypsaGlpk(PypsaBackend):
    pass
