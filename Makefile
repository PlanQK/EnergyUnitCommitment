# This makefile is used for starting optimization runs. Using various parameters that can be set, it will generate
# rules for the result files, and then generates them by starting a docker container. The possible recipes to be
# made can be found at the bottom of the file.
#
# The optimization runs for which the rules are created by searching /networks for pypsa networks and /src/config
# for config files. They can be specified using a glob expression. The result files will then consist of all combinations
# networks and configs, and a string which indicates which parameters are overwritten by the makefile and the date when
# the rule was created.

SHELL := /bin/bash
DOCKERCOMMAND := docker
DOCKERFILE = Dockerfile
DOCKERTAG = energy:1.0

# a virtual environment to run stuff without a docker container
VENV_NAME = venv

###### set problem directory ######
PROBLEMDIRECTORY := $(shell git rev-parse --show-toplevel)

###### set mount paths ######
# for development purposes, we don't build the entire image, but mount the the code that is changed often.
# We also mount the folder containing the networks and the config files
MOUNTSWEEPPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/networks/,target=/energy/Problemset
MOUNTLIBSPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/src/libs,target=/energy/libs
MOUNTCONFIGSPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/src/configs,target=/energy/configs
# only mount qpu results if there actually any results
ifeq ("$(wildcard $(PROBLEMDIRECTORY)/results_qpu_sweep)", "")
MOUNTQPURESULTPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/results_qpu_sweep,target=/energy/results_qpu
endif
MOUNTALL := $(MOUNTSWEEPPATH) $(MOUNTLIBSPATH) $(MOUNTCONFIGSPATH) $(MOUNTQPURESULTPATH)

###### define save folder ######
# choose a folder where the results should be saved to. If the folder
# doesn't exist, it will be created. If no folder is specified, one will
# be created using the name `results_general_sweep`
# If specifiying your own folder, DON'T forget '/' for a valid folder name
SAVE_FOLDER := 

###### define config file ######
# this file is the default file which contains values for all valid configurations of all solvers.
# Making will generate a run for each config file specified here. Other config files can be saved
# in /src/configs
CONFIGFILES = config-all.yaml
# CONFIGFILES = almost_empty.yaml
# CONFIGFILES = $(shell find $(PROBLEMDIRECTORY)/src/configs -name "config_[9][4-4].yaml" | sed 's!.*/!!' | sed 's!.po!!')

###### define sweep files ######
# Choose a regex that will be used to search the networks folder for networks.
# The default network is a randomly generated network containing 10 buses with
# generators that produce integer valued power and a total load of 100
NETWORKNAME = defaultnetwork.nc
NETWORKNAME = network_4qubit_2_bus.nc
# NETWORKNAME = elec_s_5.nc

# lists networks to be used using NETWORKNAME
SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/networks -name "$(strip $(NETWORKNAME))" | sed 's!.*/!!' | sed 's!.po!!')

###### define extra parameter ######
# Please check the current config-all.yaml for a list and description of all
# possible options in src/configs.
# The name of the parameter has to be stored in a variable with the
# PARAMETER_ prefix and the values for it in a variable with the
# VAL_PARAMETER_ prefix. Only Parameters which have a name with the PARAMETER_ prefix
# will be read and added to the config. You can enable/disable them by uncommenting them.
#
# Since there are multiple levels in the config dictionary the name has to
# have the following pattern, indicating all levels:
# "level1__level2__parameter". The value(s) on the other hand should be given
# as a string separated by a "__".
# E.g.	PARAMETER_KIRCHSCALEFACTOR = "ising_interface__kirchhoff__scale_factor"
# 		VAL_PARAMETER_KIRCHSCALEFACTOR = 1.0__2.0
# Comment out any parameters not currently in use.
# TODO logic for splitting up parameter values into multiple files to be made

### General Parameters
# Uncommenting a line of the form #PARMETER_* will overwrite the value in the config
# with the value below it.

### A parameter for setting the solver.
# PARAMETER_BACKEND = backend
VAL_PARAMETER_BACKEND = sqa

### Ising Model Parameters
# Determines how network, constraints, and optimization goals are encoded
# Used by any solver that uses a QUBO (sqa, dwave annealer, qaoa)

# PARAMETER_GENERATORREPRESENTATION = ising_interface__generator_representation
VAL_PARAMETER_GENERATORREPRESENTATION = with_status

# PARAMETER_LINEREPRESENTATION = ising_interface__line_representation
VAL_PARAMETER_LINEREPRESENTATION = cutpowersoftwo

# PARAMETER_KIRCHSCALEFACTOR = ising_interface__kirchhoff__scale_factor
VAL_PARAMETER_KIRCHSCALEFACTOR = 1.0

# PARAMETER_MARGINALFORMULATION = ising_interface__marginal_cost__formulation
VAL_PARAMETER_MARGINALFORMULATION = binarysplit

# PARAMETER_MONETARYSCALEFACTOR = ising_interface__marginal_cost__scale_factor
VAL_PARAMETER_MONETARYSCALEFACTOR = 0.5

# PARAMETER_OFFSETFACTOR = ising_interface__marginal_cost__offset_factor
VAL_PARAMETER_OFFSETFACTOR_VAL = 1.2


### QAOA Parameters
# PARAMETER_QAOASHOTS = qaoa_backend__shots
VAL_PARAMETER_QAOASHOTS = 500

# PARAMETER_QAOASIMULATE = qaoa_backend__simulate
VAL_PARAMETER_QAOASIMULATE = True

# PARAMETER_QAOANOISE = qaoa_backend__noise
VAL_PARAMETER_QAOANOISE = True

# PARAMETER_QAOASIMULATOR = qaoa_backend__simulator
VAL_PARAMETER_QAOASIMULATOR = aer_simulator

# PARAMETER_QAOAINITGUESS = qaoa_backend__initial_guess
VAL_PARAMETER_QAOAINITGUESS = ['rand', 'rand']

# PARAMETER_QAOAMAXITER = qaoa_backend__max_iter
VAL_PARAMETER_QAOAMAXITER = 30

# PARAMETER_QAOAREPS = qaoa_backend__repetitions
VAL_PARAMETER_QAOAREPS = 20

# PARAMETER_QAOACLASSICALOPT = qaoa_backend__classical_optimizer
VAL_PARAMETER_QAOACLASSICALOPT = COBYLA


### SQA Parameters
# PARAMETER_TRANSVERSEFIELD = sqa_backend__transverse_field_schedule
VAL_PARAMETER_TRANSVERSEFIELD = 8.0

# PARAMETER_SIQUAN_TEMP = sqa_backend__temperature_schedule
VAL_PARAMETER_SIQUAN_TEMP = 0.1

# PARAMETER_TROTTERSLICES = sqa_backend__trotter_slices
VAL_PARAMETER_TROTTERSLICES = 40__80

# PARAMETER_OPTIMIZATIONCYCLES = sqa_backend__optimization_cycles
VAL_PARAMETER_OPTIMIZATIONCYCLES = 20


### D-Wave Quantum Annealer Parameters.
# Requires an APIToken set in the config
# PARAMETER_ANNEAL_TIME = dwave_backend__annealing_time
VAL_PARAMETER_ANNEAL_TIME = 100

# PARAMETER_NUM_READS = dwave_backend__num_reads
VAL_PARAMETER_NUM_READS = 200

# PARAMETER_CHAINSTRENGTH = dwave_backend__chain_strength
VAL_PARAMETER_CHAINSTRENGTH = 60

# PARAMETER_PROGTHERMALIZATION = dwave_backend__programming_thermalization
VAL_PARAMETER_PROGTHERMALIZATION = 1

# PARAMETER_READTHERMALIZATION = dwave_backend__readout_thermalization
VAL_PARAMETER_READTHERMALIZATION = 1

# PARAMETER_STRATEGY = dwave_backend__strategy
VAL_PARAMETER_STRATEGY = lowest_energy

# PARAMETER_POSTPROCESS = dwave_backend__postprocess
VAL_PARAMETER_POSTPROCESS = flow

# PARAMETER_DWAVETIMEOUT = dwave_backend__timeout
VAL_PARAMETER_DWAVETIMEOUT = 100

# PARAMETER_SAMPLEORIGIN = dwave_backend__sample_origin
VAL_PARAMETER_SAMPLEORIGIN = filename


### GLPK Parameter
# PARAMETER_PYPSASOLVERNAME = pypsa_backend__solver_name
VAL_PARAMETER_PYPSASOLVERNAME = glpk

# PARAMETER_PYPSATIMEOUT = pypsa_backend__timeout
VAL_PARAMETER_PYPSATIMEOUT = 60

###### extra parameter string generation ######

join_with_underscore = $(subst $(eval) ,_,$(wildcard $1))

# combine each parameter name with its values, if it is not commented out
EXTRAPARAMSEPARATE = $(foreach name, $(filter PARAMETER_%,$(.VARIABLES)), \
						$(foreach value, ${VAL_${name}}, \
						$(strip ${${name}}__${value})))

# join all separate parameter___value pairs
EXTRAPARAM = $(subst " ","____",$(foreach param, \
				${EXTRAPARAMSEPARATE},$(param)))
# if no Parameters are declared in the Makefile, the string will be set to an
# empty string
ifeq ($(EXTRAPARAM),)
EXTRAPARAM = ""
endif


###### generate default SAVE_FOLDER path for current solver if no path is specified
ifeq ($(SAVE_FOLDER),)
SAVE_FOLDER = results_general_sweep/
endif

###### result files of computations ######

GENERAL_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		${SAVE_FOLDER}${filename}_${config}_${EXTRAPARAM}))


###### creating rules for result files ######
# define general target

define general
${SAVE_FOLDER}$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/networks/$(strip $(1)) .docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	$(DOCKERTAG) $(strip $(1)) $(strip $(2)) $(strip $(3))
	mkdir -p ${SAVE_FOLDER}
	mv $(PROBLEMDIRECTORY)/networks/$(strip $(1))_* ${SAVE_FOLDER}
endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extra, ${EXTRAPARAM}, \
	$(eval $(call general, ${filename}, ${config}, ${extra})))))

# end of creating rules for results



###### Define further helper targets ######

.docker.tmp: $(DOCKERFILE) src/run.py requirements.txt src/program.py src/libs/return_objects.py
	$(DOCKERCOMMAND) build -t $(DOCKERTAG) -f $(DOCKERFILE) . && touch .docker.tmp


# all plots are generated using the python plot_results script
plots: $(VENV_NAME)/bin/activate
	mkdir -p plots && . $(VENV_NAME)/bin/activate && python scripts/plot_results.py

plt: $(VENV_NAME)/bin/activate
	mkdir -p plots && . $(VENV_NAME)/bin/activate && python scripts/make_plots.py

$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || python3.9 -m venv $(VENV_NAME)
	. $(VENV_NAME)/bin/activate; python3.9 -m pip install -r requirements.txt
	touch $(VENV_NAME)/bin/activate

# rules for making a sweep for a particular solver

.PHONY: all clean plots quantumReadSweep general

all: quantumReadSweep general

general: $(GENERAL_SWEEP_FILES)

clean:
	rm -rf Problemset/info*

# create rules for make using a specific solver and setting the corresponding SAVE_FOLDER
SOLVERS = classical sqa qpu qpu_read pypsa_glpk qaoa
#TODO make rules
