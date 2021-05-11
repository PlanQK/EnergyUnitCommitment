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
    "monetaryCostFactor": 0.7,
    "kirchoffFactor": 1,
    "minUpDownFactor": 0.0,
    "trotterSlices": 32,
    "useKirchoffInequality": True,
    "dwaveAPIToken": "",
    "seed": random.randint(0, 100000),
}


errorMsg = """
Usage: run.py [classical | sqa | dwave-classical | dwave-quantum | pypsa-glpk]
Arguments:
    classical: run a classical annealing algorithm
    sqa: run the discrete time simulated quantum annealing algorithm
    dwave-classical: run the classical optimization procedure from the dwave package (tabu-search)
    dwave-quantum: run the optimization on a dwave quantum annealing device (quantum annealing)

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
    "dwave-classical": Backends.DwaveClassicalBackend,
    "dwave-tabu": Backends.DwaveTabuSampler,
    "dwave-quantum": Backends.DwaveCloudQuantumBackend,
    "pypsa-glpk": Backends.PypsaGlpk,
    "pypsa-fico": Backends.PypsaFico,
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
