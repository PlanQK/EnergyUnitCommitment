SHELL := /bin/bash
DOCKERCOMMAND := docker
REPETITIONS := 1

PROBLEMDIRECTORY := $(shell git rev-parse --show-toplevel)
# alternative in case this is not a git repository
#PROBLEMDIRECTORY := $(shell pwd)
MOUNTSWEEPPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset
MOUNTBACKENDPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/DockerInput/Backends,target=/energy/Backends
MOUNTCONFIGPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/DockerInput/Configs/config.yaml,target=/energy/config.yaml
MOUNTCONFIGSPATH := --mount type=bind,source=$(PROBLEMDIRECTORY)/DockerInput/Configs,target=/energy/Configs
MOUNTALL := $(MOUNTSWEEPPATH) $(MOUNTBACKENDPATH) $(MOUNTCONFIGSPATH)
PREFIX := infoNocostFixed


# config file
CONFIGFILES = "config.yaml"
#CONFIGFILES = $(shell find $(PROBLEMDIRECTORY)/DockerInput/Configs -name "config_[8][0-9].yaml" | sed 's!.*/!!' | sed 's!.po!!')

# general parameters
NUMBERS = $(shell seq 1 ${REPETITIONS})
TIME := $(shell date +"%Y-%m-%d_%H-%M-%S")

# sqa parameters
SIQUAN_TEMP = $(shell seq 0.1 1 0.1)
TRANSVERSE_HIGH = $(shell seq 8.0 1.0 8)
OPTIMIZATIONCYCLES = $(shell seq 5 5 20)
OPTIMIZATIONCYCLES = 1000
#OPTIMIZATIONCYCLES = 10 30 77 215 599 1668 4641 12915 35938 100000
TROTTERSLICES = $(shell seq 10 10 100)
TROTTERSLICES = 2000

# classical parameters. reuses OPTIMIZATIONCYCLES
CLASSICAL_HIGH_TEMP = $(shell seq 10.0 10.0 10)
CLASSICAL_LOW_TEMP = $(shell seq 0.5 0.5 0.5)

# dwave quantum annealer parameters. Requires an APIToken as an environmentvariabale with name
# dwaveAPIToken
ANNEAL_TIME = $(shell seq 100 50 100)
NUM_READS = $(shell seq 200 20 200)
#CHAINSTRENGTH = $(shell seq 60 30 60)
CHAINSTRENGTH = 60
SAMPLECUTSIZE = $(shell seq 200 4 200)

# Ising Model Parameters. Determines how lines are represented. Used for any solver that uses a QUBO (sqa, dwave annealer)
#PROBLEMFORMULATION = binarysplitNoMarginalCost
PROBLEMFORMULATION = fullsplitGlobalCostSquare
#PROBLEMFORMULATION = fullsplitMarginalAsPenalty
#PROBLEMFORMULATION = fullsplitMarginalAsPenaltyAverageOffset
#PROBLEMFORMULATION = fullsplitNoMarginalCost
#PROBLEMFORMULATION = fullsplitLocalMarginalEstimationDistance
#PROBLEMFORMULATION = fullsplitDirectInefficiencyPenalty

#PROBLEMFORMULATION = fullsplitMarginalAsPenalty fullsplitLocalMarginalEstimationDistance fullsplitGlobalCostSquare

MONETARYCOSTFACTOR = 0.02 0.015 0.025
#MONETARYCOSTFACTOR = 0.2 0.3 0.4

# only relevant for problem formulation using an estimation-of-marginal-costs ansatz
#OFFSETESTIMATIONFACTOR = 0.95 0.97 1.0 1.03 1.05
#ESTIMATEDCOSTFACTOR = 0.95 0.97 1.0 1.03 1.05
#OFFSETBUILDFACTOR = 0.95 0.97 1.0 1.03 1.05
# 10.0
#OFFSETESTIMATIONFACTOR = 1.32
OFFSETESTIMATIONFACTOR = 1.3805 1.3800 1.3802
#OFFSETESTIMATIONFACTOR = 1.210
# 10.1
#OFFSETESTIMATIONFACTOR = 1.349
# 60.0 by 0.16
#OFFSETESTIMATIONFACTOR = 1.3203
#OFFSETESTIMATIONFACTOR = 1.4268
#OFFSETESTIMATIONFACTOR = 1.0 1.1 1.2 1.3 1.4

ESTIMATEDCOSTFACTOR = 1.0
OFFSETBUILDFACTOR = 1.0

# glpk parameter
TIMEOUT = 60


# SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "nocostinput_15_[0]_[2][0].nc" | sed 's!.*/!!' | sed 's!.po!!')
SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "220124cost5input_[9]0_[0]_20.nc" | sed 's!.*/!!' | sed 's!.po!!')
#SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "testNetwork4QubitIsing_2_0_20.nc" | sed 's!.*/!!' | sed 's!.po!!')
# SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "testNetwork5QubitIsing_2_0_20.nc" | sed 's!.*/!!' | sed 's!.po!!')

# result files of computations



CLASSICAL_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach hightemp, ${CLASSICAL_HIGH_TEMP}, \
		$(foreach lowtemp, ${CLASSICAL_LOW_TEMP}, \
		$(foreach optimizationCycles, ${OPTIMIZATIONCYCLES}, \
		$(foreach trotterSlices, ${TROTTERSLICES}, \
		$(foreach number, ${NUMBERS}, \
		results_classical_parameter_sweep/${PREFIX}_${filename}_${hightemp}_${lowtemp}_${optimizationCycles}_${trotterSlices}_${number}))))))

SIQUAN_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach temp, ${SIQUAN_TEMP}, \
		$(foreach transverse, ${TRANSVERSE_HIGH}, \
		$(foreach optimizationCycles, ${OPTIMIZATIONCYCLES}, \
		$(foreach trotterSlices, ${TROTTERSLICES}, \
		$(foreach problemFormulation, ${PROBLEMFORMULATION}, \
		$(foreach offsetEstimationFactor, ${OFFSETESTIMATIONFACTOR}, \
		$(foreach estimatedCostFactor, ${ESTIMATEDCOSTFACTOR}, \
		$(foreach offsetBuildFactor, ${OFFSETBUILDFACTOR}, \
		$(foreach monetaryCostFactor, ${MONETARYCOSTFACTOR}, \
		$(foreach number, ${NUMBERS}, \
		results_sqa_sweep/${PREFIX}_${filename}_${temp}_${transverse}_${optimizationCycles}_${trotterSlices}_${problemFormulation}_${offsetEstimationFactor}_${estimatedCostFactor}_${offsetBuildFactor}_${monetaryCostFactor}_${number})))))))))))

QUANTUM_ANNEALING_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach anneal_time, ${ANNEAL_TIME}, \
		$(foreach num_reads, ${NUM_READS}, \
		$(foreach problemFormulation, ${PROBLEMFORMULATION}, \
		$(foreach chain_strength, ${CHAINSTRENGTH}, \
		$(foreach offsetEstimationFactor, ${OFFSETESTIMATIONFACTOR}, \
		$(foreach estimatedCostFactor, ${ESTIMATEDCOSTFACTOR}, \
		$(foreach offsetBuildFactor, ${OFFSETBUILDFACTOR}, \
		$(foreach monetaryCostFactor, ${MONETARYCOSTFACTOR}, \
		$(foreach number, ${NUMBERS}, \
		results_qpu_sweep/${PREFIX}_${filename}_${anneal_time}_${num_reads}_${problemFormulation}_${chain_strength}_${offsetEstimationFactor}_${estimatedCostFactor}_${offsetBuildFactor}_${monetaryCostFactor}_${number}))))))))))

QUANTUM_ANNEALING_READ_RESULTS = $(foreach filename, $(SWEEPFILES), \
		$(foreach anneal_time, ${ANNEAL_TIME}, \
		$(foreach num_reads, ${NUM_READS}, \
		$(foreach problemFormulation, ${PROBLEMFORMULATION}, \
		$(foreach chain_strength, ${CHAINSTRENGTH}, \
		$(foreach sampleCutSize, ${SAMPLECUTSIZE}, \
		$(foreach number, ${NUMBERS}, \
		results_qpu_read_sweep/${PREFIX}_${filename}_${anneal_time}_${num_reads}_${problemFormulation}_${chain_strength}_${sampleCutSize}_${number})))))))

GLPK_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach timeout, $(TIMEOUT), \
		$(foreach number, ${NUMBERS}, \
		results_pypsa_glpk_sweep/${PREFIX}_${filename}_${timeout}_${number})))

QAOA_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach timeout, $(TIMEOUT), \
		$(foreach number, ${NUMBERS}, \
		$(foreach time, ${TIME}, \
		$(foreach config, ${CONFIGFILES}, \
		results_qaoa_sweep/${PREFIX}_${filename}_${timeout}_${number}_${time}_${config})))))


## creating rules for result files


# define classical parameter sweep targets

define classicalParameterSweep
results_classical_sweep/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	--env temperatureSchedule=[$(strip $(2)),$(strip $(3))] \
	--env transverseFieldSchedule=[0] \
	--env optimizationCycles=$(strip $(4)) \
	--env trotterSlices=$(strip $(5)) \
	--env outputInfo=${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 classical
	mkdir -p results_classical_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6)) results_classical_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach hightemp, ${CLASSICAL_HIGH_TEMP}, \
	$(foreach lowtemp, ${CLASSICAL_LOW_TEMP}, \
	$(foreach optimizationCycles, ${OPTIMIZATIONCYCLES}, \
	$(foreach trotterSlices, ${TROTTERSLICES}, \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call classicalParameterSweep, ${filename}, ${hightemp}, ${lowtemp}, ${optimizationCycles}, ${trotterSlices}, ${number}))))))))


# define siquan sweep targets

define sqaParameterSweep
results_sqa_sweep/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8))_$(strip $(9))_$(strip $(10))_$(strip $(11)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	--env temperatureSchedule=[$(strip $(2)),iF,0.0001] \
	--env transverseFieldSchedule=[$(strip $(3)),0.0] \
	--env optimizationCycles=$(strip $(4)) \
	--env trotterSlices=$(strip $(5)) \
	--env problemFormulation=$(strip $(6)) \
	--env offsetEstimationFactor=$(strip $(7)) \
	--env estimatedCostFactor=$(strip $(8)) \
	--env offsetBuildFactor=$(strip $(9)) \
	--env monetaryCostFactor=$(strip $(10)) \
	--env outputInfo=${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8))_$(strip $(9))_$(strip $(10))_$(strip $(11)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 sqa
	mkdir -p results_sqa_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8))_$(strip $(9))_$(strip $(10))_$(strip $(11)) results_sqa_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach temp, ${SIQUAN_TEMP}, \
	$(foreach transverseField, ${TRANSVERSE_HIGH}, \
	$(foreach optimizationCycles, ${OPTIMIZATIONCYCLES}, \
	$(foreach trotterSlices, ${TROTTERSLICES}, \
	$(foreach problemFormulation, ${PROBLEMFORMULATION}, \
	$(foreach offsetEstimationFactor, ${OFFSETESTIMATIONFACTOR}, \
	$(foreach estimatedCostFactor, ${ESTIMATEDCOSTFACTOR}, \
	$(foreach offsetBuildFactor, ${OFFSETBUILDFACTOR}, \
	$(foreach monetaryCostFactor, ${MONETARYCOSTFACTOR}, \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call sqaParameterSweep, ${filename}, ${temp}, ${transverseField}, ${optimizationCycles}, ${trotterSlices}, \
			${problemFormulation}, ${offsetEstimationFactor}, ${estimatedCostFactor}, ${offsetBuildFactor}, ${monetaryCostFactor}, ${number})))))))))))))

#define D-Wave qpu targets

define qpuParameterSweep
results_qpu_sweep/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8))_$(strip $(9))_$(strip $(10)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	--env annealing_time=$(strip $(2)) \
	--env num_reads=$(strip $(3)) \
	--env problemFormulation=$(strip $(4)) \
	--env chain_strength=$(strip $(5)) \
	--env offsetEstimationFactor=$(strip $(6)) \
	--env estimatedCostFactor=$(strip $(7)) \
	--env offsetBuildFactor=$(strip $(8)) \
	--env monetaryCostFactor=$(strip $(9)) \
	--env outputInfo=${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8))_$(strip $(9))_$(strip $(10)) \
	--env inputNetwork=$(strip $(1)) \
	--env dwaveAPIToken=$(dwaveAPIToken) \
	energy:1.0 dwave-qpu
	mkdir -p results_qpu_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8))_$(strip $(9))_$(strip $(10)) results_qpu_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach anneal_time, ${ANNEAL_TIME}, \
	$(foreach num_reads, ${NUM_READS}, \
	$(foreach problemFormulation, ${PROBLEMFORMULATION}, \
	$(foreach chain_strength, ${CHAINSTRENGTH}, \
	$(foreach offsetEstimationFactor, ${OFFSETESTIMATIONFACTOR}, \
	$(foreach estimatedCostFactor, ${ESTIMATEDCOSTFACTOR}, \
	$(foreach offsetBuildFactor, ${OFFSETBUILDFACTOR}, \
	$(foreach monetaryCostFactor, ${MONETARYCOSTFACTOR}, \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call qpuParameterSweep, ${filename}, ${anneal_time}, ${num_reads}, ${problemFormulation}, ${chain_strength}, \
			${offsetEstimationFactor}, ${estimatedCostFactor}, ${offsetBuildFactor}, ${monetaryCostFactor}, ${number}))))))))))))


# define targets for old sample data

define qpuReadSweep
results_qpu_read_sweep/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7)): docker.tmp \
		results_qpu_sweep/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(7))
	$(DOCKERCOMMAND) run $(MOUNTALL) \
			--mount type=bind,source=$(PROBLEMDIRECTORY)/results_qpu_sweep/,target=/energy/results_qpu \
	--env annealing_time=$(strip $(2)) \
	--env num_reads=$(strip $(3)) \
	--env problemFormulation=$(strip $(4)) \
	--env chain_strength=$(strip $(5)) \
	--env sampleCutSize=$(strip $(6)) \
	--env outputInfo=${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7)) \
	--env inputNetwork=$(strip $(1)) \
	--env inputInfo=results_qpu/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(7)) \
	energy:1.0 dwave-read-qpu
	mkdir -p results_qpu_read_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7)) results_qpu_read_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach anneal_time, ${ANNEAL_TIME}, \
	$(foreach num_reads, ${NUM_READS}, \
	$(foreach problemFormulation, ${PROBLEMFORMULATION}, \
	$(foreach chain_strength, ${CHAINSTRENGTH}, \
	$(foreach sampleCutSize, ${SAMPLECUTSIZE}, \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call qpuReadSweep, ${filename}, ${anneal_time}, ${num_reads}, ${problemFormulation}, ${chain_strength}, ${sampleCutSize}, ${number})))))))))


# define glpk targets

define pypsa-glpk
results_pypsa_glpk_sweep/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	--env outputInfo=${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3)) \
	--env inputNetwork=$(strip $(1)) \
	--env timeout=$(strip $(2)) \
	energy:1.0 pypsa-glpk
	mkdir -p results_pypsa_glpk_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3)) results_pypsa_glpk_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach timeout, $(TIMEOUT), \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call pypsa-glpk, ${filename}, ${timeout}, ${number})))))

# define qaoa target

define qaoa
results_qaoa_sweep/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run $(MOUNTALL) \
	--env outputInfo=${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5)) \
	--env inputNetwork=$(strip $(1)) \
	--env timeout=$(strip $(2)) \
	energy:1.0 qaoa $(strip $(5))
	mkdir -p results_qaoa_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/${PREFIX}_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))* results_qaoa_sweep/
	mv $(PROBLEMDIRECTORY)/sweepNetworks/Qaoa_$(strip $(1))_$(strip $(4))_$(strip $(5))* results_qaoa_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach timeout, $(TIMEOUT), \
	$(foreach number, ${NUMBERS}, \
	$(foreach time, ${TIME}, \
	$(foreach config, ${CONFIGFILES}, \
	$(eval $(call qaoa, ${filename}, ${timeout}, ${number}, ${time}, ${config})))))))

# end of creating rules for results



# Define further helper targets

docker.tmp: Dockerfile DockerInput/run.py DockerInput/requirements.txt
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

all: classicalParSweep siquanParSweep quantumAnnealParSweep pypsa-glpk

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
