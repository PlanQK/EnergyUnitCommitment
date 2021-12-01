import time

from .BackendBase import BackendBase
import pypsa
from EnvironmentVariableManager import EnvironmentVariableManager

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

        print(self.metaInfo["solution"]["state"] )
        return solution


    def transformProblemForOptimizer(self, network):
        print("transforming problem...") 
        self.network = network.copy()
        self.network.generators.committable = True
        self.network.generators.p_nom_extendable = False

        # avoid committing a generator and setting output to 0 
        self.network.generators_t.p_min_pu = self.network.generators_t.p_max_pu

        # add slack generators which have to be used last. penalty has to be
        # significantly higher than using a regular generator
        maximum_loads_t = self.network.loads_t['p_set'].max()
        for load in maximum_loads_t.index:
            bus = self.network.loads.loc[load]['bus']
            self.network.add("Generator",f"pos_slack_{bus}",bus=bus,
                    committable=False,
                    p_min_pu=0.0,
                    p_max_pu= maximum_loads_t[load],
                    marginal_cost=self.slack_gen_penalty,
                    min_down_time=0,
                    start_up_cost=0,
                    p_nom=1)
            self.network.add("Generator",f"neg_slack_{bus}",bus=bus,
                    committable=False,
                    p_min_pu= - maximum_loads_t[load],
                    p_max_pu=0.0,
                    marginal_cost= - self.slack_gen_penalty,
                    min_down_time=0,
                    start_up_cost=0,
                    p_nom=1)

        self.model = pypsa.opf.network_lopf_build_model(self.network,
                self.network.snapshots,
                formulation="kirchhoff")
        self.opt = pypsa.opf.network_lopf_prepare_solver(self.network,
                solver_name=self.solver_name)
        return self.model


    def transformSolutionToNetwork(self, network, transformedProblem, solution):
        print("Writing from pyoyo model to network is not implemented")
        
        print(self.metaInfo["solution"]["genStates"] )
        print(self.metaInfo["solution"]["lineValues"] )

        for snapshot in network.snapshots:
            exceeded_demand = 0
            missing_demand = 0
            totalCost = 0
            for bus in network.buses.index:
                cur_neg_slack = self.model.generator_p.get_values()[(f"neg_slack_{bus}",snapshot)]
                exceeded_demand += cur_neg_slack
                totalCost += cur_neg_slack ** 2
                if cur_neg_slack:
                    print(f"EXCEEDED LOAD AT {bus}: {cur_neg_slack}")

                cur_pos_slack = self.model.generator_p.get_values()[(f"pos_slack_{bus}",snapshot)]
                missing_demand += cur_pos_slack 
                totalCost += cur_pos_slack ** 2
                if cur_pos_slack:
                    print(f"MISSING LOAD AT {bus}: {cur_pos_slack}")

                if not (cur_neg_slack + cur_pos_slack):
                    print(f"OPTIMAL LOAD AT {bus}")

            dev_from_optimum_gen = exceeded_demand + missing_demand
            if dev_from_optimum_gen == 0:
                print(f"POWER GENERATION MATCHES TOTAL LOAD IN {snapshot}")
            else:
                print(f"TOTAL POWER/LOAD IMBALANCE {dev_from_optimum_gen}")
            self.metaInfo["totalCost"] = totalCost
            print(f"TOTAL COST: {totalCost}")
        return

    def optimize(self, transformedProblem):
        print("starting optimization...")

        sol = self.opt.solve(transformedProblem)

        solverstring = str(sol["Solver"])
        solvingTime = solverstring.splitlines()[-1].split()[1]
        self.metaInfo["time"] = solvingTime

        sol.write()

        committed_gen = []
        for key in self.model.generator_status.get_values().keys():
            if self.model.generator_status.get_values()[key] == 1.0:
                committed_gen.append(key[0])
        self.metaInfo["status"] = committed_gen

        return self.model

    def getMetaInfo(self):
        return self.metaInfo

    
    def __init__(self, solver_name = "glpk", slack_gen_penalty = 100.0):
        self.metaInfo = {}
        self.solver_name = solver_name
        self.slack_gen_penalty = slack_gen_penalty
        envMgr = EnvironmentVariableManager()
        self.kirchhoffFactor = float(envMgr["kirchhoffFactor"])
        self.monetaryCostFactor = float(envMgr["monetaryCostFactor"])
        self.minUpDownFactor = float(envMgr["minUpDownFactor"])
        self.slackVarFactor = float(envMgr["slackVarFactor"])


class PypsaFico(PypsaBackend):

    def __init__(self):
        super().__init__("fico")

class PypsaGlpk(PypsaBackend):

    def __init__(self):
        super().__init__("glpk")
