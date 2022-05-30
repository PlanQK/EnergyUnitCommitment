SHELL := /bin/bash
DOCKERCOMMAND := docker
#DOCKERFILE = PlanQK_Dockerfile
DOCKERFILE = Dockerfile
DOCKERTAG = energy:1.0

###### set problem directory ######
PROBLEMDIRECTORY := $(shell git rev-parse --show-toplevel)
# alternative in case this is not a git repository
#PROBLEMDIRECTORY := $(shell pwd)

###### set mount paths ######
MOUNTSWEEPPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset
MOUNTBACKENDPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/src/libs/Backends,target=/energy/libs/Backends
MOUNTCONFIGSPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/src/Configs,target=/energy/Configs
MOUNTALL := $(MOUNTSWEEPPATH) $(MOUNTBACKENDPATH) $(MOUNTCONFIGSPATH)

###### define save folder ######
# choose a folder where the results should be saved to. If the folder
# doesn't exist, it will be created. Common choices are:
# results_classical_sweep, results_sqa_sweep, results_qpu_sweep,
# results_qpu_read_sweep, results_pypsa_glpk_sweep, results_qaoa_sweep
# DON'T forget the "/" after the folder name
SAVE_FOLDER := "results_qaoa_sweep/"

###### define config file ######
CONFIGFILES = "config.yaml"
#CONFIGFILES = $(shell find $(PROBLEMDIRECTORY)/src/Configs -name "config_[9][4-4].yaml" | sed 's!.*/!!' | sed 's!.po!!')

###### define sweep files ######
#SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "220124cost5input_[1]0_[0]_20.nc" | sed 's!.*/!!' | sed 's!.po!!')
SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "testNetwork4QubitIsing_2_0_20.nc" | sed 's!.*/!!' | sed 's!.po!!')
#SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "testNetwork5QubitIsing_2_0_20.nc" | sed 's!.*/!!' | sed 's!.po!!')

###### define extra parameter ######
# Please check the current config-all.yaml for a list and description of all
# possible options.
# The name of the parameter has to be stored in a variable with the
# PARAMETER_ prefix and the values for it in a variable with the
# VAL_PARAMETER_ prefix.
# Since there are multiple levels in the config dictionary the name has to
# have the following pattern, indicating all levels:
# "level1__level2__parameterName". The value(s) on the other hand should be given
# as a string separated by a "__".
# E.g.	PARAMETER_KIRCHSCALEFACTOR = "IsingInterface__kirchhoff__scaleFactor"
# 		VAL_PARAMETER_KIRCHSCALEFACTOR = "1.0__5.0__10.0"
# Comment out any parameters not currently in use.


### General Parameters
#PARAMETER_BACKEND = "Backend"
VAL_PARAMETER_BACKEND = "qaoa"


### Ising Model Parameters
# Determines how network, constraints, and optimization goals are encoded
# Used by any solver that uses a QUBO (sqa, dwave annealer, qaoa)

#PARAMETER_FORMULATION = \
	"IsingInterface__formulation"
VAL_PARAMETER_FORMULATION = "binarysplit"

PARAMETER_KIRCHSCALEFACTOR = \
	"IsingInterface__kirchhoff__scaleFactor"
VAL_PARAMETER_KIRCHSCALEFACTOR = "1.0__5.0__10.0"

#PARAMETER_KIRCHFACTOR = \
	"IsingInterface__kirchhoff__kirchhoffFactor"
VAL_PARAMETER_KIRCHFACTOR = "1.0__1.1"

#PARAMETER_MARGINALFORMULATION = \
	"IsingInterface__marginalCost__formulation"
VAL_PARAMETER_MARGINALFORMULATION = "binarysplit"

#PARAMETER_MONETARYCOSTFACTOR = \
	"IsingInterface__marginalCost__monetaryCostFactor"
VAL_PARAMETER_MONETARYCOSTFACTOR = "0.2__0.3__0.4"

#PARAMETER_MONETARYSCALEFACTOR = \
	"IsingInterface__marginalCost__scaleFactor"
VAL_PARAMETER_MONETARYSCALEFACTOR = "1.0__5.0__10.0"

#PARAMETER_OFFSETESTIMATIONFACTOR = \
	"IsingInterface__marginalCost__offsetEstimationFactor"
VAL_PARAMETER_OFFSETESTIMATIONFACTOR_VAL = "1.1__1.2__1.3"

#PARAMETER_ESTIMATEDCOSTFACTOR = \
	"IsingInterface__marginalCost__estimatedCostFactor"
VAL_PARAMETER_ESTIMATEDCOSTFACTOR_VAL = "1.0"

#PARAMETER_OFFSETBUILDFACTOR = \
	"IsingInterface__marginalCost__offsetBuildFactor"
VAL_PARAMETER_OFFSETBUILDFACTOR_VAL = "1.0"

#PARAMETER_MINUPDOWNFACTOR = \
	"IsingInterface__minUpDownTime__minUpDownFactor"
VAL_PARAMETER_MINUPDOWNFACTOR = "1.0"


### QAOA Parameters
#PARAMETER_QAOASHOTS = \
	"QaoaBackend__shots"
VAL_PARAMETER_QAOASHOTS = "500"

#PARAMETER_QAOASIMULATE = \
	"QaoaBackend__simulate"
VAL_PARAMETER_QAOASIMULATE = "True"

#PARAMETER_QAOANOISE = \
	"QaoaBackend__noise"
VAL_PARAMETER_QAOANOISE = "True"

#PARAMETER_QAOASIMULATOR = \
	"QaoaBackend__simulator"
VAL_PARAMETER_QAOASIMULATOR = "aer_simulator"

#PARAMETER_QAOAINITGUESS = \
	"QaoaBackend__initial_guess"
#TODO: Parse a list to run.py
VAL_PARAMETER_QAOAINITGUESS = "[rand rand]"

#PARAMETER_QAOAMAXITER = \
	"QaoaBackend__max_iter"
VAL_PARAMETER_QAOAMAXITER = "100"

#PARAMETER_QAOAREPS = \
	"QaoaBackend__repetitions"
VAL_PARAMETER_QAOAREPS = "50"

#PARAMETER_QAOACLASSICALOPT = \
	"QaoaBackend__classical_optimizer"
VAL_PARAMETER_QAOACLASSICALOPT = "COBYLA"


### sqa parameters
SIQUAN_TEMP = ""
SIQUAN_TEMP_VAL = $(shell seq 0.1 1 0.1)
TRANSVERSE_HIGH = ""
TRANSVERSE_HIGH_VAL = $(shell seq 8.0 1.0 8)
OPTIMIZATIONCYCLES = "optimizationCycles"
OPTIMIZATIONCYCLES_VAL = $(shell seq 5 5 20)
OPTIMIZATIONCYCLES_VAL = 100
#OPTIMIZATIONCYCLES_VAL = 10 30 77 215 599 1668 4641 12915
TROTTERSLICES = "trotterSlices"
TROTTERSLICES_VAL = $(shell seq 10 10 100)
TROTTERSLICES_VAL = 500

### classical parameters. reuses OPTIMIZATIONCYCLES
CLASSICAL_HIGH_TEMP = ""
CLASSICAL_HIGH_TEMP_VAL = $(shell seq 10.0 10.0 10)
CLASSICAL_LOW_TEMP = ""
CLASSICAL_LOW_TEMP_VAL = $(shell seq 0.5 0.5 0.5)

### dwave quantum annealer parameters. Requires an APIToken as an environmentvariabale with name
### dwaveAPIToken
ANNEAL_TIME = "annealing_time"
ANNEAL_TIME_VAL = $(shell seq 100 50 100)
NUM_READS = "num_reads"
NUM_READS_VAL = $(shell seq 200 20 200)
CHAINSTRENGTH = "chain_strength"
#CHAINSTRENGTH_VAL = $(shell seq 60 30 60)
CHAINSTRENGTH_VAL = 60
SAMPLECUTSIZE = "sampleCutSize"
SAMPLECUTSIZE_VAL = $(shell seq 200 4 200)


### glpk parameter
TIMEOUT = "timeout"
TIMEOUT_VAL = 60

###### extra parameter string generation ######

EXTRAPARAMSEPARATE = 	$(foreach name, $(filter PARAMETER_%,$(.VARIABLES)), \
						$(foreach value, ${VAL_${name}}, \
						${${name}}___${value}))

EXTRAPARAM = 	$(subst " ","____",$(foreach param, \
				${EXTRAPARAMSEPARATE},$(param)))

ifeq ($(EXTRAPARAM),)
EXTRAPARAM = ""
endif
###### result files of computations ######

GENERAL_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		${SAVE_FOLDER}${filename}_${config}_${EXTRAPARAM}))

QUANTUM_ANNEALING_READ_RESULTS = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		$(foreach extraparam, ${EXTRAPARAM}, \
		${SAVE_FOLDER}${filename}_${config}_${extraparam})))

###### creating rules for result files ######

# define targets for old sample data

define qpuReadSweep
${SAVE_FOLDER}$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	--mount type=bind,source=$(PROBLEMDIRECTORY)/results_qpu_sweep,target=/energy/results_qpu \
	$(DOCKERTAG) $(strip $(1)) $(strip $(2)) $(strip $(3))
	mkdir -p ${SAVE_FOLDER}
	mv $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1))_dwave-read-qpu* ${SAVE_FOLDER}

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extraparam, ${EXTRAPARAM}, \
	$(eval $(call qpuReadSweep, ${filename}, ${config}, ${extraparam})))))

# define general target

define general
${SAVE_FOLDER}$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	$(DOCKERTAG) $(strip $(1)) $(strip $(2)) $(strip $(3))
	mkdir -p ${SAVE_FOLDER}
	mv $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1))_* ${SAVE_FOLDER}

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extra, ${EXTRAPARAM}, \
	$(eval $(call general, ${filename}, ${config}, ${extra})))))

# end of creating rules for results



###### Define further helper targets ######

docker.tmp: $(DOCKERFILE) src/run.py requirements.txt src/program.py
	$(DOCKERCOMMAND) build -t $(DOCKERTAG) -f $(DOCKERFILE) . && touch docker.tmp


# all plots are generated using the python plot_results script
plots: venv/bin/activate
	mkdir -p plots && source venv/bin/activate && python plot_results.py

plt: venv/bin/activate
	mkdir -p plots && source venv/bin/activate && python makePlots.py

venv/bin/activate:
	python3 -m venv venv && pip install -r requirements.txt && pip install

# rules for making a sweep for a particular solver

.PHONY: all clean plots quantumReadSweep general

all: quantumReadSweep general

general: $(GENERAL_SWEEP_FILES)

# requires old data to be reused or dwaveAPIToken
quantumReadSweep: $(QUANTUM_ANNEALING_READ_RESULTS)

clean:
	rm -rf Problemset/info*
