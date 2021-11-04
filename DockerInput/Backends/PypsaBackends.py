import time

from .BackendBase import BackendBase
import pypsa
from EnvironmentVariableManager import EnvironmentVariableManager


class PypsaBackend(BackendBase):
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
        for snapshot in network.snapshots:
            exceeded_demand = 0
            missing_demand = 0
            for bus in network.buses.index:
                exceeded_demand -= self.model.generator_p.get_values()[(f"neg_slack_{bus}",snapshot)]
                missing_demand += self.model.generator_p.get_values()[(f"pos_slack_{bus}",snapshot)]
            if not (exceeded_demand == 0 or missing_demand == 0):
                print("error in slack generators")
            else:
                dev_from_optimum = exceeded_demand + missing_demand
                if dev_from_optimum == 0:
                    print(f"all loads met in {snapshot}")
                else:
                    print(f"missing load in {snapshot}: {dev_from_optimum}")
        return

    def optimize(self, transformedProblem):
        print("starting optimization...")
        tic = time.perf_counter()
        self.opt.solve(transformedProblem).write()
        self.metaInfo["time"] = time.perf_counter() - tic

        committed_gen = []
        for key in self.model.generator_status.get_values().keys():
            if self.model.generator_status.get_values()[key] == 1.0:
                committed_gen.append(key[0])
        self.metaInfo["status"] = committed_gen
        #self.metaInfo["output"] = self.model.generator_p.get_values()


        return self.model

    def getMetaInfo(self):
        return self.metaInfo

    
    def __init__(self, solver_name = "glpk", slack_gen_penalty = 100.0):
        self.metaInfo = {}
        self.solver_name = solver_name
        self.slack_gen_penalty = slack_gen_penalty
 
    def read_envMgr(self):
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
