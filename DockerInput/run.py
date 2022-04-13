"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""

import sys
import json, yaml
import random
import pypsa
import Backends
from Backends.Adapter import Adapter
from EnvironmentVariableManager import EnvironmentVariableManager


FOLDER = "Problemset"

DEFAULT_ENV_VARIABLES = {
    "inputNetwork": "input.nc",
    "inputInfo": "",
    "outputNetwork": "",
    "outputInfo": "output.json",
    "optimizationCycles": 1000,
    "temperatureSchedule": "[0.1,iF,0.0001]",
    "transverseFieldSchedule": "[10,.1]",
    "monetaryCostFactor": 0.4,
    "kirchhoffFactor": 1.0,
    "slackVarFactor": 60.0,
    "minUpDownFactor": 0.0,
    "trotterSlices": 32,
    "problemFormulation": "fullsplitNoMarginalCost",
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
    "timeout": "-1",
    "maxOrder": 0,
    "sampleCutSize": 200,
    "threshold": 0.5,
    "offsetEstimationFactor" : 1.0,
    "estimatedCostFactor" : 1.0,
    "offsetBuildFactor" : 1.0,
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
    dwave-read-qpu: reuse the optimization of a dwave quantum annealing device (read from local drive)


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
    "dwave-read-qpu": Backends.DwaveReadQPU,
    "qaoa": Backends.QaoaQiskit,
#    "qaoa": Backends.QaoaQiskitIsing,
}

def main():
    assert sys.argv[1] in ganBackends.keys(), errorMsg

    if len(sys.argv) == 2:
        # mock object
        inputData = "config-sqa.yaml"
    else:
        inputData = sys.argv[2]

    adapter = Adapter.makeAdapter(inputData)

    # TODO currently used so makefile rules still work by adding extra configs to the adapter from
    # environment. Remove this later
    envMgr = EnvironmentVariableManager(DEFAULT_ENV_VARIABLES)
    variablesNotInAdapter = {key : value 
                    for key, value in envMgr.returnEnvironmentVariables().items()
                    if key in DEFAULT_ENV_VARIABLES.keys() and key not in adapter.config
                    }
    adapter.config = {**variablesNotInAdapter, **adapter.config}
    # end filling up with environmentvariables

    OptimizerClass = ganBackends[sys.argv[1]]

    optimizer = OptimizerClass(adapter)

    pypsaNetwork = adapter.getNetwork()

    # validate input has to throw and catch exceptions on it's own
    optimizer.validateInput("Problemset", adapter.config['inputNetwork']))

    transformedProblem = optimizer.transformProblemForOptimizer(pypsaNetwork)

    solution = optimizer.optimize(transformedProblem)

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
        json.dump(optimizer.getMetaInfo(), write_file, indent=2, default=str)
    return


if __name__ == "__main__":
    main()
