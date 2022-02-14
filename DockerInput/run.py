"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""

import sys
import json, yaml
import random
import pypsa
import Backends
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
    "monetaryCostFactor": 0.1,
    "kirchhoffFactor": 1.0,
    "slackVarFactor": 70.0,
    "minUpDownFactor": 0.0,
    "trotterSlices": 32,
    "problemFormulation": "fullsplitMarginalAsPenalty",
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
    "threshold": 0.5
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
}


def main():
    assert len(sys.argv) == 3, errorMsg
    assert sys.argv[1] in ganBackends.keys(), errorMsg

    with open(sys.argv[2]) as file:
        config = yaml.safe_load(file)

    #TODO: move away from env_variables and pass direct to functions??
    #TODO: use config.yaml to set env_variables??
    DEFAULT_ENV_VARIABLES["problemFormulation"] = config["IsingInterface"]["problemFormulation"]

    # Create Singleton object for the first time with the default parameters
    envMgr = EnvironmentVariableManager(DEFAULT_ENV_VARIABLES)

    config["QaoaBackend"]["outputInfoTime"] = envMgr["outputInfoTime"]

    OptimizerClass = ganBackends[sys.argv[1]]

    optimizer = OptimizerClass(config=config)
    try:
        optimizer.validateInput("Problemset", str(envMgr['inputNetwork']))
    except TypeError:
        print("network has been blacklisted for this optimizer and timeout value")
        print("abort optimization")
        return

    pypsaNetwork = pypsa.Network(f"Problemset/{str(envMgr['inputNetwork'])}")
    transformedProblem = optimizer.transformProblemForOptimizer(pypsaNetwork)

#    try:
    solution = optimizer.optimize(transformedProblem)
#    except ValueError:
#        optimizer.handleOptimizationStop("Problemset",str(envMgr['inputNetwork']))
#        print("stopping optimization during solving")
#       return

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
