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
        optimization to metaInfo dictionary. No further postprocessing
        for pypsa is done
        """
        # value of bits corresponding to generators in network
        self.metaInfo["solution"] = {}
        self.metaInfo["solution"]["genStates"] = {
                gen[0] : value for gen,value in self.model.generator_status.get_values().items()
        }
        # list of indices of active generators
        self.metaInfo["solution"]["state"] = [
                idx for idx in range(len(network.generators))
                if self.metaInfo["solution"]["genStates"][network.generators.index[idx]] == 1.0
        ]
        # 
        self.metaInfo["solution"]["lineValues"] = {
                line[1] : value 
                for line,value in self.model.passive_branch_p.get_values().items()
        }

        self.metaInfo["postprocessingTime"] = 0.0
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
                solver_name=self.metaInfo["pypsaBackend"]["solver_name"])
        self.opt.options["tmlim"] = self.metaInfo["timeout"]
        return self.model


    def transformSolutionToNetwork(self, network, transformedProblem, solution):
        print("Writing from pyoyo model to network is not implemented")

        if self.metaInfo["pypsaBackend"]["terminationCondition"] == "infeasible":
            print("no feasible solution, stop writing to network")

        elif self.metaInfo["pypsaBackend"]["terminationCondition"] != "infeasible":
            
            print(self.metaInfo["solution"]["genStates"] )
            print(self.metaInfo["solution"]["lineValues"] )

            totalCost = 0
            costPerSnapshot = {snapshot : 0.0 for snapshot in self.network.snapshots}

            for key, val in  self.model.generator_p.get_values().items():
                incurredCost = self.network.generators["marginal_cost"].loc[key[0]] * val
                totalCost += incurredCost
                costPerSnapshot[key[1]] += incurredCost
            self.metaInfo["totalCost"] = totalCost
            self.metaInfo["marginalCost"] = totalCost
            self.metaInfo["powerImbalance"] = 0.0
            self.metaInfo["kirchhoffCost"] = 0.0
            print(f"Total marginal cost: {self.metaInfo['marginalCost']}")
        return


    def optimize(self, transformedProblem):
        print("starting optimization...")

        sol = self.opt.solve(transformedProblem)

        solverstring = str(sol["Solver"])
        solvingTime = solverstring.splitlines()[-1].split()[1]
        terminationCondition = solverstring.splitlines()[-7].split()[2]
        self.metaInfo["optimizationTime"] = solvingTime
        self.metaInfo["pypsaBackend"]["terminationCondition"] = terminationCondition

        sol.write()

        committed_gen = []
        for key in self.model.generator_status.get_values().keys():
            if self.model.generator_status.get_values()[key] == 1.0:
                committed_gen.append(key[0])
        self.metaInfo["status"] = committed_gen
        return self.model


    def getMetaInfo(self):
        return self.metaInfo

    
    def __init__(self, config: dict, solver_name = "glpk", ):
        super().__init__(config=config)
        self.metaInfo["pypsaBackend"]["solver_name"] = solver_name

        if self.metaInfo["timeout"] < 0:
            self.metaInfo["timeout"] = 1000


class PypsaFico(PypsaBackend):

    def __init__(self, config: dict):
        super().__init__(config=config, solver_name="fico")

class PypsaGlpk(PypsaBackend):

    def __init__(self, config: dict):
        super().__init__(config=config, solver_name="glpk")
