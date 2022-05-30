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

### general parameters
PARAMETER_BACKEND = "Backend"
VAL_PARAMETER_BACKEND = "qaoa"

### Ising Model Parameters.
# Determines how network, constraints, and optimization goals are encoded
# Used by any solver that uses a QUBO (sqa, dwave annealer, qaoa)
# TODO: add comments for options that exist

# network representation:
# 	line encoding
# constraints:
# 	kirchhoff
# 	minUpDownTime
# optimization goals
# 	marginalCost

PARAMETER_FORMULATION = "IsingInterface_formulation"
VAL_PARAMETER_FORMULATION = "binarysplit"

MONETARYCOSTFACTOR = "monetaryCostFactor"
MONETARYCOSTFACTOR_VAL = 0.2 0.3 0.4

# only relevant for problem formulation using an estimation-of-marginal-costs ansatz
OFFSETESTIMATIONFACTOR = "offsetEstimationFactor"
OFFSETESTIMATIONFACTOR_VAL = 1.1 1.2 1.3

ESTIMATEDCOSTFACTOR = "estimatedCostFactor"
ESTIMATEDCOSTFACTOR_VAL = 1.0

OFFSETBUILDFACTOR = "offsetBuildFactor"
OFFSETBUILDFACTOR_VAL = 1.0

PARAMETER_SCALEFACTOR = "IsingInterface_kirchhoff_scaleFactor"
#PARAMETER_SCALEFACTOR_VAL = 1.5
VAL_PARAMETER_SCALEFACTOR = "1.0_5.0_10.0"

PARAMETER_KIRCHFACTOR = "IsingInterface_kirchhoff_kirchhoffFactor"
#PARAMETER_KIRCHFACTOR_VAL = 1.5
VAL_PARAMETER_KIRCHFACTOR = "1.0_1.1"

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


### Example extra parameter string generation
TEST_PARAM = "test" # parameter1 name
TEST_PARAM_VAL = $(shell seq 5 5 10) # parameter1 values
ANOTHER_TEST_PARAM = "test2" # parameter2 name
ANOTHER_TEST_PARAM_VAL = 1.0 1.1 # parameter2 values
# create single string from all extra parameters ('_' between parameters & '-' between parameter name and its value)
#EXTRAPARAM = 	$(foreach value1, $(TEST_PARAM_VAL), \
				$(foreach value2, $(ANOTHER_TEST_PARAM_VAL), \
				${TEST_PARAM}-${value1}_${ANOTHER_TEST_PARAM}-${value2}))

### extra parameter generation
#EXTRAPARAM = 	$(foreach value1, $(PARAMETER_SCALEFACTOR_VAL), \
				$(foreach value2, $(PARAMETER_KIRCHFACTOR_VAL), \
				${PARAMETER_SCALEFACTOR}-${value1}_${PARAMETER_KIRCHFACTOR}-${value2}))

EXTRAPARAMSEPARATE = 	$(foreach name, $(filter PARAMETER_%,$(.VARIABLES)), \
						$(foreach value, ${VAL_${name}}, \
						${${name}}__${value}))

EXTRAPARAM = 	$(subst " ","___",$(foreach param, ${EXTRAPARAMSEPARATE},\
				$(param)))

#BACKEND = sqa
#EXTRAPARAM = Backend-$(strip $(BACKEND))

###### result files of computations ######

#GENERAL_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		$(foreach extraparam, ${EXTRAPARAM}, \
		${SAVE_FOLDER}${filename}_${config}_${extraparam})))

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
	$(DOCKERTAG) $(strip $(1)) $(strip $(2)) $(strip $(3)) $(strip $(4))
	mkdir -p ${SAVE_FOLDER}
	mv $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1))_* ${SAVE_FOLDER}

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extranames, ${EXTRAPARAM}, \
	$(eval $(call general, ${filename}, ${config}, ${extranames})))))

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
