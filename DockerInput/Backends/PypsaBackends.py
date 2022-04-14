import time

from .BackendBase import BackendBase
import pypsa

class PypsaBackend(BackendBase):

    def validateInput(self, path, network):
        pass

    def handleOptimizationStop(self, path, network):
        pass

    def processSolution(self, network, transformedProblem, solution):
        """
        writes results of generator states and line values of pypsa
        optimization to output dictionary. No further postprocessing
        for pypsa is done
        """
        # value of bits corresponding to generators in network
        self.output["results"]["genStates"] = {
                gen[0] : value for gen,value in self.model.generator_status.get_values().items()
        }
        # list of indices of active generators
        self.output["results"]["state"] = [
                idx for idx in range(len(network.generators))
                if self.output["results"]["genStates"][network.generators.index[idx]] == 1.0
        ]
        # 
        self.output["results"]["lineValues"] = {
                line[1] : value 
                for line,value in self.model.passive_branch_p.get_values().items()
        }

        self.output["results"]["postprocessingTime"] = 0.0
        return solution


    def transformProblemForOptimizer(self, network):
        print("transforming problem...") 
        self.network = network.copy()
        self.network.generators.committable = True
        self.network.generators.p_nom_extendable = False

        # avoid committing a generator and setting output to 0 
        self.network.generators_t.p_min_pu = self.network.generators_t.p_max_pu
        self.model = pypsa.opf.network_lopf_build_model(self.network,
                self.network.snapshots,
                formulation="kirchhoff")
        self.opt = pypsa.opf.network_lopf_prepare_solver(self.network,
                solver_name=self.reader.config["PypsaBackend"]["solver_name"])
        self.opt.options["tmlim"] = self.reader.config["PypsaBackend"]["timeout"]
        return self.model


    def transformSolutionToNetwork(self, network, transformedProblem, solution):
        print("Writing from pyoyo model to network is not implemented")

        if self.reader.config["PypsaBackend"]["terminationCondition"] == "infeasible":
            print("no feasible solution, stop writing to network")

        elif self.reader.config["PypsaBackend"]["terminationCondition"] != "infeasible":
            
            print(self.output["results"]["genStates"] )
            print(self.output["results"]["lineValues"] )

            totalCost = 0
            costPerSnapshot = {snapshot : 0.0 for snapshot in self.network.snapshots}

            for key, val in  self.model.generator_p.get_values().items():
                incurredCost = self.network.generators["marginal_cost"].loc[key[0]] * val
                totalCost += incurredCost
                costPerSnapshot[key[1]] += incurredCost
            self.output["results"]["totalCost"] = totalCost
            self.output["results"]["marginalCost"] = totalCost
            self.output["results"]["powerImbalance"] = 0.0
            self.output["results"]["kirchhoffCost"] = 0.0
            print(f"Total marginal cost: {self.output['results']['marginalCost']}")
        return


    def optimize(self, transformedProblem):
        print("starting optimization...")

        sol = self.opt.solve(transformedProblem)

        solverstring = str(sol["Solver"])
        solvingTime = solverstring.splitlines()[-1].split()[1]
        terminationCondition = solverstring.splitlines()[-7].split()[2]
        self.output["results"]["optimizationTime"] = solvingTime
        self.reader.config["PypsaBackend"]["terminationCondition"] = terminationCondition

        sol.write()

        committed_gen = []
        for key in self.model.generator_status.get_values().keys():
            if self.model.generator_status.get_values()[key] == 1.0:
                committed_gen.append(key[0])
        self.output["results"]["status"] = committed_gen
        return self.model


    def getMetaInfo(self):
        return self.metaInfo

    
    def __init__(self, *args):
        super().__init__(args)

        if self.reader.config["PypsaBackend"]["timeout"] < 0:
            self.reader.config["PypsaBackend"]["timeout"] = 1000


class PypsaFico(PypsaBackend):

    def __init__(self, *args):
        super().__init__(args, solver_name="fico")

class PypsaGlpk(PypsaBackend):

    def __init__(self, *args):
        super().__init__(args, solver_name="glpk")
