"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""

import sys
import json
import random
import pypsa
import Backends
from EnvironmentVariableManager import EnvironmentVariableManager

FOLDER = "Problemset"

DEFAULT_ENV_VARIABLES = {
    "inputNetwork": "input.nc",
    "outputNetwork": "",
    "outputInfo": "output.json",
    "optimizationCycles": 1000,
    "temperatureSchedule": "[0.1,iF,0.0001]",
    "transverseFieldSchedule": "[10,.1]",
    "monetaryCostFactor": 0.0,
    "kirchhoffFactor": 1.0,
    "slackVarFactor": 70.0,
    "minUpDownFactor": 0.0,
    "trotterSlices": 32,
    "dwaveAPIToken": "",
    "dwaveBackend": "hybrid_discrete_quadratic_model_version1",
    "annealing_time": 500,
    "programming_thermalization": 0,
    "readout_thermalization": 0,
    "num_reads": 1,
    "chain_strength": 250,
    "seed": random.randint(0, 100000),
    "strategy": "LowestEnergy",
    "lineRepresentation": 0,
    "postprocess": "flow",
    "timeout": "50",
}


errorMsg = """
Usage: run.py [classical | sqa | dwave-tabu | dwave-greedy | dwave-hybrid | dwave-qpu]
Arguments:
    classical: run a classical annealing algorithm
    sqa: run the discrete time simulated quantum annealing algorithm
    dwave-tabu: run the classical optimization procedure locally using the dwave package (tabu-search)
    dwave-greedy: run the classical optimization procedure locally using the dwave package (greedy)
    dwave-hybrid: run a hybrid optimization procedure from dwave (through cloud)
    dwave-qpu: run the optimization on a dwave quantum annealing device (through cloud)


Any further settings are specified through environment variables:
    optimizationCycles: 1000  Number of optimization cycles
            (each spin gets an update proposal once during one cycle)
    temperatureSchedule: []  (for sqa) how the temperature changes during the annealing run
    transverseFieldSchedule: [] (for sqa) how the transverse field changes during the annealing run
    cubicConstraints: false  DWave does not support cubic constraints

"""

ganBackends = {
    "classical": Backends.ClassicalBackend,
    "sqa": Backends.SqaBackend,
    "dwave-tabu": Backends.DwaveTabuSampler,
    "dwave-greedy": Backends.DwaveSteepestDescent,
    "pypsa-glpk": Backends.PypsaGlpk,
    "pypsa-fico": Backends.PypsaFico,
    "dwave-hybrid": Backends.DwaveCloudHybrid,
    "dwave-qpu": Backends.DwaveCloudDirectQPU,
}


def main():
    # Create Singleton object for the first time with the default parameters
    envMgr = EnvironmentVariableManager(DEFAULT_ENV_VARIABLES)

    assert len(sys.argv) == 2, errorMsg
    assert sys.argv[1] in ganBackends.keys(), errorMsg

    OptimizerClass = ganBackends[sys.argv[1]]
    optimizer = OptimizerClass()
    try:
        optimizer.validateInput("Problemset", str(envMgr['inputNetwork']))
    except ValueError:
        print("network has been blacklisted for this optimizer and timeout value")
        print("stopping optimization")
        return

    pypsaNetwork = pypsa.Network(f"Problemset/{str(envMgr['inputNetwork'])}")
    transformedProblem = optimizer.transformProblemForOptimizer(pypsaNetwork)

    try:
        solution = optimizer.optimize(transformedProblem)
    except ValueError:
        optimizer.handleOptimizationStop("Problemset",str(envMgr['inputNetwork']))
        print("stopping optimization")
        return

    processedSolution = optimizer.processSolution(
        pypsaNetwork, transformedProblem, solution
    )
    outputNetwork = optimizer.transformSolutionToNetwork(
        pypsaNetwork, transformedProblem, processedSolution
    )
    if envMgr["outputNetwork"]:
        outputNetwork.export_to_netcdf(
            f"Problemset/{str(envMgr['outputNetwork'])}"
        )

    with open(f"Problemset/{str(envMgr['outputInfo'])}", "w") as write_file:
        json.dump(optimizer.getMetaInfo(), write_file, indent=2)
    return


if __name__ == "__main__":
    main()
