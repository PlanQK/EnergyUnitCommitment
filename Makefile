SHELL := /bin/bash
DOCKERCOMMAND := docker
REPETITIONS := 1

PROBLEMDIRECTORY := $(shell git rev-parse --show-toplevel)
# alternative in case this is not a git repository
#PROBLEMDIRECTORY := $(shell pwd)
MOUNTSWEEPPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset
MOUNTBACKENDPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/src/libs/Backends,target=/energy/libs/Backends
MOUNTCONFIGPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/src/Configs/config.yaml,target=/energy/config.yaml
MOUNTCONFIGSPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/src/Configs,target=/energy/Configs
MOUNTALL := $(MOUNTSWEEPPATH) $(MOUNTBACKENDPATH) $(MOUNTCONFIGSPATH)
PREFIX := infoNocostFixed

###### general parameters ######
NUMBERS = $(shell seq 1 ${REPETITIONS})
TIME := $(shell date +"%Y-%m-%d_%H-%M-%S")

###### define config file ######
CONFIGFILES = "config.yaml"
#CONFIGFILES = $(shell find $(PROBLEMDIRECTORY)/src/Configs -name "config_[9][4-4].yaml" | sed 's!.*/!!' | sed 's!.po!!')

###### define sweep files ######
# SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "nocostinput_15_[0]_[2][0].nc" | sed 's!.*/!!' | sed 's!.po!!')
#SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "220124cost5input_[9]0_[0]_20.nc" | sed 's!.*/!!' | sed 's!.po!!')
SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "testNetwork4QubitIsing_2_0_20.nc" | sed 's!.*/!!' | sed 's!.po!!')
# SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "testNetwork5QubitIsing_2_0_20.nc" | sed 's!.*/!!' | sed 's!.po!!')

###### define extra parameter ######

### sqa parameters
SIQUAN_TEMP = ""
SIQUAN_TEMP_VAL = $(shell seq 0.1 1 0.1)
TRANSVERSE_HIGH = ""
TRANSVERSE_HIGH_VAL = $(shell seq 8.0 1.0 8)
OPTIMIZATIONCYCLES = "optimizationCycles"
OPTIMIZATIONCYCLES_VAL = $(shell seq 5 5 20)
OPTIMIZATIONCYCLES_VAL = 1000
#OPTIMIZATIONCYCLES_VAL = 10 30 77 215 599 1668 4641 12915 35938 100000
TROTTERSLICES = "trotterSlices"
TROTTERSLICES_VAL = $(shell seq 10 10 100)
TROTTERSLICES_VAL = 2000

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

### Ising Model Parameters. Determines how lines are represented. Used for any solver that uses a QUBO (sqa, dwave annealer)
PROBLEMFORMULATION = "formulation"
PROBLEMFORMULATION_VAL = binarysplitNoMarginalCost
#PROBLEMFORMULATION_VAL = fullsplitGlobalCostSquare
#PROBLEMFORMULATION_VAL = fullsplitMarginalAsPenalty
#PROBLEMFORMULATION_VAL = fullsplitMarginalAsPenaltyAverageOffset
#PROBLEMFORMULATION_VAL = fullsplitNoMarginalCost
#PROBLEMFORMULATION_VAL = fullsplitLocalMarginalEstimationDistance
#PROBLEMFORMULATION_VAL = fullsplitDirectInefficiencyPenalty
#PROBLEMFORMULATION_VAL = fullsplitMarginalAsPenalty fullsplitLocalMarginalEstimationDistance fullsplitGlobalCostSquare

MONETARYCOSTFACTOR = "monetaryCostFactor"
MONETARYCOSTFACTOR_VAL = 0.02 0.015 0.025
#MONETARYCOSTFACTOR_VAL = 0.2 0.3 0.4

# only relevant for problem formulation using an estimation-of-marginal-costs ansatz
OFFSETESTIMATIONFACTOR = "offsetEstimationFactor"
#OFFSETESTIMATIONFACTOR_VAL = 0.95 0.97 1.0 1.03 1.05
# 10.0
#OFFSETESTIMATIONFACTOR_VAL = 1.32
OFFSETESTIMATIONFACTOR_VAL = 1.3805 1.3800 1.3802
#OFFSETESTIMATIONFACTOR_VAL = 1.210
# 10.1
#OFFSETESTIMATIONFACTOR_VAL = 1.349
# 60.0 by 0.16
#OFFSETESTIMATIONFACTOR_VAL = 1.3203
#OFFSETESTIMATIONFACTOR_VAL = 1.4268
#OFFSETESTIMATIONFACTOR_VAL = 1.0 1.1 1.2 1.3 1.4

ESTIMATEDCOSTFACTOR = "estimatedCostFactor"
ESTIMATEDCOSTFACTOR_VAL = 1.0
#ESTIMATEDCOSTFACTOR_VAL = 0.95 0.97 1.0 1.03 1.05

OFFSETBUILDFACTOR = "offsetBuildFactor"
#OFFSETBUILDFACTOR_VAL = 0.95 0.97 1.0 1.03 1.05
OFFSETBUILDFACTOR_VAL = 1.0

SCALEFACTOR = "scaleFactor"
SCALEFACTOR_VAL = 2.0

KIRCHFACTOR = "kirchhoffFactor"
KIRCHFACTOR_VAL = 1.5

### glpk parameter
TIMEOUT = "timeout"
TIMEOUT_VAL = 60


### Example extra parameter string generation
PARAM1 = 'test' # parameter1 name
PARAM1VAL = $(shell seq 5 5 10) # parameter1 values
PARAM2 = 'test2' # parameter2 name
PARAM2VAL = 1.0 1.1 # parameter2 values
# create single string from all extra parameters ('_' between parameters & '-' between parameter name and its value)
EXTRAPARAM = 	$(foreach value1, $(PARAM1VAL), \
				$(foreach value2, $(PARAM2VAL), \
				${PARAM1}-${value1}_${PARAM2}-${value2}))

### extra parameter generation
EXTRAPARAM = 	$(foreach value1, $(SCALEFACTOR_VAL), \
				$(foreach value2, $(KIRCHFACTOR_VAL), \
				${SCALEFACTOR}-${value1}_${KIRCHFACTOR}-${value2}))
EXTRAPARAM = ''



###### result files of computations ######

CLASSICAL_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		$(foreach extraparam, ${EXTRAPARAM}, \
		results_classical_parameter_sweep/${filename}_${config}_${extraparam})))

SIQUAN_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		$(foreach extraparam, ${EXTRAPARAM}, \
		results_sqa_sweep/${filename}_${config}_${extraparam})))

QUANTUM_ANNEALING_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		$(foreach extraparam, ${EXTRAPARAM}, \
		results_qpu_sweep/${filename}_${config}_${extraparam})))

QUANTUM_ANNEALING_READ_RESULTS = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		$(foreach extraparam, ${EXTRAPARAM}, \
		results_qpu_read_sweep/${filename}_${config}_${extraparam})))

GLPK_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		$(foreach extraparam, ${EXTRAPARAM}, \
		results_pypsa_glpk_sweep/${filename}_${config}_${extraparam})))

QAOA_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		$(foreach extraparam, ${EXTRAPARAM}, \
		results_qaoa_sweep/${filename}_${config}_${extraparam})))


###### creating rules for result files ######

# define classical parameter sweep targets

define classicalParameterSweep
results_classical_sweep/$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	energy:1.0 $(strip $(2)) $(strip $(1)) $(strip $(3))
	mkdir -p results_classical_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1))_classical* results_classical_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extraparam, ${EXTRAPARAM}, \
	$(eval $(call classicalParameterSweep, ${filename}, ${config}, ${extraparam})))))

# define siquan sweep targets

define sqaParameterSweep
results_sqa_sweep/$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	energy:1.0 $(strip $(2)) $(strip $(1)) $(strip $(3))
	mkdir -p results_sqa_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1))_sqa* results_sqa_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extraparam, ${EXTRAPARAM}, \
	$(eval $(call sqaParameterSweep, ${filename}, ${config}, ${extraparam})))))

#define D-Wave qpu targets

define qpuParameterSweep
results_qpu_sweep/$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	energy:1.0 $(strip $(2)) $(strip $(1)) $(strip $(3))
	mkdir -p results_qpu_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1))_dwave-qpu* results_qpu_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extraparam, ${EXTRAPARAM}, \
	$(eval $(call qpuParameterSweep, ${filename}, ${config}, ${extraparam})))))

# define targets for old sample data

define qpuReadSweep
results_qpu_read_sweep/$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	energy:1.0 $(strip $(2)) $(strip $(1)) $(strip $(3))
	mkdir -p results_qpu_read_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1))_dwave-read-qpu* results_qpu_read_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extraparam, ${EXTRAPARAM}, \
	$(eval $(call qpuReadSweep, ${filename}, ${config}, ${extraparam})))))

# define glpk targets

define pypsa-glpk
results_pypsa_glpk_sweep/$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	energy:1.0 $(strip $(2)) $(strip $(1)) $(strip $(3))
	mkdir -p results_pypsa_glpk_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1))_pypsa-glpk* results_pypsa_glpk_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extraparam, ${EXTRAPARAM}, \
	$(eval $(call pypsa-glpk, ${filename}, ${config}, ${extraparam})))))

# define qaoa target

define qaoa
results_qaoa_sweep/$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	energy:1.0 $(strip $(2)) $(strip $(1)) $(strip $(3))
	mkdir -p results_qaoa_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1))_qaoa* results_qaoa_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach config, ${CONFIGFILES}, \
	$(foreach extraparam, ${EXTRAPARAM}, \
	$(eval $(call qaoa, ${filename}, ${config}, ${extraparam})))))

# end of creating rules for results



###### Define further helper targets ######

docker.tmp: Dockerfile src/run.py src/requirements.txt src/program.py
	$(DOCKERCOMMAND) build -t energy:1.0 . && touch docker.tmp


# all plots are generated using the python plot_results script
plots: venv/bin/activate
	mkdir -p plots && source venv/bin/activate && python plot_results.py

plt: venv/bin/activate
	mkdir -p plots && source venv/bin/activate && python makePlots.py

venv/bin/activate:
	python3 -m venv venv && pip install -r requirements.txt && pip install 

# rules for making a sweep for a particular solver

.PHONY: all clean classicalParSweep siquanParSweep quantumAnnealParSweep plots pypsa-glpk quantumReadSweep

all: classicalParSweep siquanParSweep quantumAnnealParSweep pypsa-glpk qaoa

classicalParSweep: $(CLASSICAL_PARAMETER_SWEEP_FILES)

siquanParSweep: $(SIQUAN_PARAMETER_SWEEP_FILES)

# requires a dWaveAPIToken
quantumAnnealParSweep: $(QUANTUM_ANNEALING_SWEEP_FILES)

# requires old data to be reused or dwaveAPIToken
quantumReadSweep: $(QUANTUM_ANNEALING_READ_RESULTS)

pypsa-glpk: $(GLPK_SWEEP_FILES)

qaoa: $(QAOA_SWEEP_FILES)

clean:
	rm -rf Problemset/info*
