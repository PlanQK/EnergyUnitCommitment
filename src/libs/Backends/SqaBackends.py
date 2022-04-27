from ast import literal_eval

# try import from local .so
# Error message for image: herrd1/siquan:latest 
# /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by /energy/libs/Backends/siquan.cpython-39-x86_64-linux-gnu.so)`
try:
    from . import siquan
# try import from installed module siquan
except ImportError:
    import siquan


from .InputReader import InputReader
from .IsingPypsaInterface import IsingBackbone
from .BackendBase import BackendBase
import time


class ClassicalBackend(BackendBase):
    def __init__(self, reader: InputReader):
        super().__init__(reader=reader)
        self.solver = siquan.DTSQA()

        # mock > remove later
        self.metaInfo = {"SqaBackend":{}}

    def validateInput(self, path, network):
        pass

    def handleOptimizationStop(self, path, network):
        pass

    def processSolution(self, network, transformedProblem, solution):
        self.output["results"]["postprocessingTime"] = 0.0
        return solution

    def transformProblemForOptimizer(self, network):
        print("transforming problem...")
        return IsingBackbone.buildIsingProblem(
                network,
                config=self.config["IsingInterface"]
                )
        return IsingPypsaInterface.buildCostFunction(
            network,
        )

    def transformSolutionToNetwork(self, network, transformedProblem, solution):
        self.printResults(transformedProblem, solution)
        # transformedProblem.addSQASolutionToNetwork(
        #     network, solution["state"]
        # )
        return network

    def optimize(self, transformedProblem):
        print("starting optimization...")
        self.configureSolver(HSchedule="[0]")
        tic = time.perf_counter()
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        self.output["results"]["optimizationTime"] = time.perf_counter() - tic
        # parse the entry in "state" before using it
        result["state"] = literal_eval(result["state"])
        self.writeResultsToOutput(result, transformedProblem)
        print("done")
        return result

    def configureSolver(self, 
            HSchedule = None,
            TSchedule = None,
            trotterSlices = None,
            steps = None
            ):
        """
        reads and sets sqa solver parameter from the environment unless
        a hyper parameter is set in the function call
        
        Args:
            HSchedule: (str) a string describing the transverse field schedule
            TSchedule: (str) a string describing the temperature schedule
            trotterSlices: (int) number of trotter slices to be used
            steps: (int) number of steps to be used
        Returns:
            (None) modifies self.solver and sets hyperparameters
        """
        try:
            self.solver.setSeed(self.config["SqaBackend"]["seed"])
        except KeyError:
            pass
        self.solver.setHSchedule(HSchedule or self.config["transverseFieldSchedule"])
        self.solver.setTSchedule(TSchedule or self.config["temperatureSchedule"])
        self.solver.setTrotterSlices(trotterSlices or int(self.config["trotterSlices"]))
        self.solver.setSteps(steps or int(self.config["optimizationCycles"]))
        return

    def printResults(self, transformedProblem, solution):
        """
        print the solution and then several derived values of the solution to quantify how
        good it is
        
        Args:
            transformedProblem: (IsingPypsaInterface) the isinginterface instance that encoded the spin glass
            problem
            solution: (list) list of all qubits which have spin -1 in the solution

        Returns:
            (None) Only side effect is printing information
        """
        print(f"\n--- Solution ---")
#        print(f"Qubits with spin -1: {solution['state']}")
#        print(f"Power on transmission lines: {transformedProblem.getLineValues(solution['state'])}")
        print(f"\n--- Meta parameters of the solution ---")
        print(f"Cost at each bus: {transformedProblem.individualCostContribution(solution['state'])}")
        print(f"Total Kirchhoff cost: {self.output['results']['kirchhoffCost']}")
        print(f"Total power imbalance: {self.output['results']['powerImbalance']}")
        print(f"Marginal Cost at each bus: {transformedProblem.individualMarginalCost(solution['state'])}")
        print(f"Total Power generated: {transformedProblem.calcTotalPowerGenerated(solution['state'])}")
        print(f"Total marginal cost: {self.output['results']['marginalCost']}")
        print(
            f"Total cost (with constant terms): {self.output['results']['totalCost']}\n"
        )
        return
    
    def writeResultsToOutput(self, result, transformedProblem):
        """
        This writes solution specific values of the optimizer result and the ising spin glass
        problem solution the output dictionary. Parse the value to the key "state" via
        literal_eval before calling this function.
        
        solution:
            result: (dict) the python dictionary returned from the sqa solver
            transformedProblem: (IsingPypsaInterface) the isinginterface instance that encoded 
                    the problem into an ising sping glass problem
        Returns:
            (None) modifies self.output with solution specific parameters and values
        """
        for key in result:
            self.output["results"][key] = result[key]
        self.output["results"]["totalCost"] = transformedProblem.calcCost(result["state"])
        self.output["results"]["solution"] = transformedProblem.calcMarginalCost(result["state"])
        self.output["results"]["kirchhoffCost"] = transformedProblem.calcKirchhoffCost(result["state"])
        self.output["results"]["powerImbalance"] = transformedProblem.calcPowerImbalance(result["state"])
        self.output["results"]["marginalCost"] = transformedProblem.calcMarginalCost(result["state"])
        self.output["results"]["SqaBackend"]["individualCost"] = transformedProblem.individualCostContribution(
                result["state"]
        )
        self.output["results"]["SqaBackend"]["eigenValues"] = sorted(transformedProblem.getHamiltonianEigenvalues()[0])
        self.output["results"]["SqaBackend"]["hamiltonian"] = transformedProblem.getHamiltonianMatrix()


class SqaBackend(ClassicalBackend):
    def optimize(self, transformedProblem):
        print("starting optimization...")
        self.configureSolver()
        tic = time.perf_counter()
        result = self.solver.minimize(
            transformedProblem.siquanFormat(),
            transformedProblem.numVariables(),
        )
        self.output["results"]["optimizationTime"] = time.perf_counter() - tic
        # parse the entry in "state" before using it
        result["state"] = literal_eval(result["state"])
        self.writeResultsToOutput(result, transformedProblem)
        print("done")
        return result
