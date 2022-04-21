"""This file is the entrypoint for the docker run command.
The docker container loads the pypsa model and performs the optimization of the unit commitment problem.
"""

import sys
import random
import libs.Backends as Backends
from program import run


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
    "postprocess": "flow",
    "timeout": "-1",
    "sampleCutSize": 200,
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
    "test": Backends.QaoaQiskit
}

def main():
    assert sys.argv[1] in ganBackends.keys(), errorMsg

    if len(sys.argv) == 3:
        inputData = sys.argv[2]
        network = sys.argv[3]
        param = None
    else:
        inputData = sys.argv[2]
        network = sys.argv[3]
        param = sys.argv[4]

    extraParams = {}
    if param is not None:
        paramList = param.split("_")
        for item in paramList:
            splitItem = item.split("-")
            extraParams[splitItem[0]] = splitItem[1]

    run(data=network, params=inputData, storeFile=True, extraParams=extraParams)

    return


if __name__ == "__main__":
    main()
