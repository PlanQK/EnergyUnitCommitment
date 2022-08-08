## About

The problem at hand to be solved is optimizing the dispatch and transmission flow over time in an energy grid. We focus on a simplified version
which focuses on deciding which generators to commit to meet the demand. This repository provides classical and quantum-based containerized solvers in python to
solve the unit commitment problem given by a [PyPSA](https://github.com/PyPSA/PyPSA) network. It also contains the code to provide it as service on 
the [PlanQK platform](https://platform.planqk.de). 


## How to use

You can use this repository by making various recipes in the makefile. An optimization run can be started by making the recipe `make general`, which will 
start a docker container. Various settings can be adjusted in either the makefile or by providing a config file. We provide an example network and config 
file which can be used right away. The recipe will start an optimization run for every combination of network, config file and setting found by in the makefile. 
The result of the optimization will be saved as a JSON file.

This repository also contains some auxiliary scripts to generate problem instance and analyze results. Using those will be explained later.


### Available solvers
Currently, we support the following solvers for the unit commitment problem:

1. Mixed Integer Linear Programming using GLPK. The linear programm is obtained via PyPSA.
2. Simulated Quantum Annealing. More information on this solver can be found [here](https://github.com/PlanQK/SimulatedQuantumAnnealing)
3. QAOA via IBM's [Qiskit](https://qiskit.org) runtime. This is extremly limited in problemsize. Some parts require an API-Token
4. D-Wave's cloud solvers. These require an API-Token and include quantum annealing, a hybrid solver and tabu search.


### Adjusting the Makefile

The makefile contains variables that specify where the PyPSA network to be solved is stored, which config file to use, and can
also be used to temporarily overwrite values in the config file. A brief overview of the parameters you can adjust are the following

1. `CONFIGFILES` : This contains a glob which will be used to search the folder `./src/configs/` for config files.
2. `NETWORKNAME` : This contains a glob which will be used to search the folder `./networks/` for PyPSA networks.
3. `SAVE_FOLDER` : This specifies a path relative to the repositiories's root where the results will be saved to.

Overwriting values of the config files is further explained in the Makefile. Additional information on the variables above  can also be found in the makefile.
When cloning the repositiory, they are set up to point to an example network, the template for new config files, using the simulated quatum annealing solver
and save it to `results_sqa_sweep`.

### Configuring a solver

Which solver and which settings are used can be specified in a yaml file. The file `./src/configs/config-all.yaml` contains information on all possible
parameters that can be set for the various solvers, the structure that the config file has to have and the possibe values for each parameter. 


## Additional uses
### Plotting results
We also provide some functionality to aggregate result saved in json files as pandas data frames and generate plots based on them. You can do so by using the 
`PlotAgent` to read results, and `make_figure` to generate a plot based on that data. Further information can be found in the doc strings. The file 
`makePlotsExample.py` explains how to do and provides a few examples. You can also run the recipe `make plt` to execute the script `makePlots.py`. This script
will be ignored by git so you can copy and rename the example file.

For qaoa results, `statAnalysis.py`, and `plotQaoa.py` provide methods to aggregate the results of runs. (TODO: Unify all modules that aggregate and plot results).


### Generating test networks
The script `problemGenerator.py` generates random problem instances of the unit commitment problem. You can speficy the number of busses and the average capacity
of a transmission line when calling it. These networks will be written to `networks`. (TODO: Change it to write to `networks` and not `Problemset`)
The script `input/convert_to_json.py` can be used to convert pypsa networks into json files, which is the format used by the planqk service.


### Testing PlankQK services
The folder `scripts` contains two short scripts to build the image similar to the one build by the platform. Further information can be found in these scripts. In order to
build the service on the platform, you have to use `PlanQK_requirements.txt` and rename it as `requirements.txt`. You also have to include the pypsa zip file
because the current version on pip doesn't have the patch for reading jsons yet. When testing it with qaoa, keep in mind that it will take forever if the network is not tiny.
Right now, testing it with qaoa doesn't write the result properly (TODO: fix that). The result that is returned by the service when testing it locally, will be written to the 
file `JobResponse` at the top level of the repository.

## Documentation
You can find documentation for D-Waves cloud solvers [here](https://docs.ocean.dwavesys.com/en/stable/index.html). For QAOA, you can find the algorithm 
[here](https://qiskit.org/textbook/ch-applications/qaoa.html) and the documentation for qiskit [here](https://qiskit.org/documentation).
You can find further information on the PlanQK platform and [PyPSA](https://github.com/PyPSA/PyPSA) github page.

You can generate further documentation using the doc strings. (TODO: generate documentation for the siquan module)

For more information on the solvers, you can have a look at `DESCRIPTION.md`, which is the description of the service on the platform and `UseCaseDescription.md`, which gives a
more detailed introduction in the optimization problem that we try to solve.

## Known Issues

TODO

## Contributing
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

