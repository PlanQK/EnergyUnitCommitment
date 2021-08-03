"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""

import sys
import json
import random
import pypsa
import Backends
from EnvironmentVariableManager import EnvironmentVariableManager

INPUT_NETWORK = "Problemset/input.nc"
OUTPUT_NETWORK = "Problemset/output.nc"

DEFAULT_ENV_VARIABLES = {
    "optimizationCycles": 1000,
    "temperatureSchedule": "[10,iF,0.001]",
    "transverseFieldSchedule": "[10,0]",
    "monetaryCostFactor": 0.0,
    "kirchhoffFactor": 1.0,
    "slackVarFactor": 10.0,
    "minUpDownFactor": 0.0,
    "trotterSlices": 32,
    "dwaveAPIToken": "",
    "dwaveBackend": "hybrid_discrete_quadratic_model_version1",
    "seed": random.randint(0, 100000),
}


errorMsg = """
Usage: run.py [classical | sqa | dwave-classical | dwave-quantum | pypsa-glpk]
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
    pypsaNetwork = pypsa.Network(INPUT_NETWORK)
    optimizer = OptimizerClass()
    transformedProblem = optimizer.transformProblemForOptimizer(pypsaNetwork)
    solution = optimizer.optimize(transformedProblem)
    outputNetwork = optimizer.transformSolutionToNetwork(
        pypsaNetwork, transformedProblem, solution
    )
    outputNetwork.export_to_netcdf(OUTPUT_NETWORK)

    with open(
        f"Problemset/{optimizer.__class__.__name__}.json", "w"
    ) as write_file:
        json.dump(optimizer.getMetaInfo(), write_file, indent=2)
    return


if __name__ == "__main__":
    main()
