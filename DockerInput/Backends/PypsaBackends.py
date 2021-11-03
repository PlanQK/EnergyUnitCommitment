import time

from .BackendBase import BackendBase
import pypsa
from EnvironmentVariableManager import EnvironmentVariableManager


class PypsaBackend(BackendBase):
    def transformProblemForOptimizer(self, network):
        print("transforming problem...") 
        self.network = network 
        self.network.generators.committable = True
        self.network.generators.p_nom_extendable = False
        # avoid committing a generator and setting output to 0 
        for name in self.network.generators.index:
            self.network.generators_t.p_min_pu[name] = 1.0
        self.model = pypsa.opf.network_lopf_build_model(self.network,
                self.network.snapshots,
                formulation="kirchhoff")
        

        self.opt = pypsa.opf.network_lopf_prepare_solver(self.network,
                solver_name=self.solver_name)
        return self.model

    def transformSolutionToNetwork(self, network, transformedProblem, solution):
        print("transforming Problem...")
        self.read_envMgr()
        solution.generator_status.pprint()
        return

    def optimize(self, transformedProblem):
        print("starting optimization...")
        tic = time.perf_counter()
        self.opt.solve(transformedProblem).write()
        self.metaInfo["time"] = time.perf_counter() - tic

        # write info

        for gen, time in self.model.generator_status_index:
            self.metaInfo["runtime_sec"] = 0
            self.metaInfo["state"] = 0

        return self.model

    def getMetaInfo(self):
        return self.metaInfo

    
    def __init__(self, solver_name = "glpk"):
        self.metaInfo = {}
        self.solver_name = solver_name
 
    def read_envMgr(self):
        envMgr = EnvironmentVariableManager()
        self.kirchhoffFactor = float(envMgr["kirchhoffFactor"])
        self.monetaryCostFactor = float(envMgr["monetaryCostFactor"])
        self.minUpDownFactor = float(envMgr["minUpDownFactor"])
        self.slackVarFactor = float(envMgr["slackVarFactor"])
                

# add additional pyomo constraints:
# flow have to be an integer
# disable optimizing power flow
def extra_functionality(network, snapshots):
    pass


class PypsaFico(PypsaBackend):

    def __init__(self):
        super().__init__("fico")

class PypsaGlpk(PypsaBackend):

    def __init__(self):
        super().__init__("glpk")
