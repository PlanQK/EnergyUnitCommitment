SHELL := /bin/bash
DOCKERCOMMAND := docker
REPETITIONS := 1

PROBLEMDIRECTORY := $(shell git rev-parse --show-toplevel)
# alternative in case this is not a git repository
#PROBLEMDIRECTORY := $(shell pwd)

# general parameters
NUMBERS = $(shell seq 1 ${REPETITIONS})

# sqa parameters
SIQUAN_TEMP = $(shell seq 0.1 1 0.1)
TRANSVERSE_HIGH = $(shell seq 10.0 1.0 10)
OPTIMIZATIONCYCLES = 1000
#OPTIMIZATIONCYCLES = 10 30 77 215 599 1668 4641 12915 35938 100000

# classical parameters. reuses OPTIMIZATIONCYCLES
CLASSICAL_HIGH_TEMP = $(shell seq 10.0 10.0 10)
CLASSICAL_LOW_TEMP = $(shell seq 0.5 0.5 0.5)

# dwave quantum annealer parameters. Requires an APIToken as an environmentvariabale with name
# dwaveAPIToken
ANNEAL_TIME = $(shell seq 110 50 110)
NUM_READS = $(shell seq 365 20 365)
SLACKVARFACTOR = $(shell seq 30 10 30)
CHAINSTRENGTH = 80
SAMPLECUTSIZE = $(shell seq 100 4 100)

# Ising Model Parameters. Determines how lines are represented. Used for any solver that uses a QUBO (sqa, dwave annealer)
LINEREPRESENTATION = $(shell seq 0 1 0)
MAXORDER = 1

# glpk parameter
TIMEOUT = 30


SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "nocostinput_1[4-4]_[0-9]_[2][0].nc" | sed 's!.*/!!' | sed 's!.po!!')

# result files of computations

CLASSICAL_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach hightemp, ${CLASSICAL_HIGH_TEMP}, \
		$(foreach lowtemp, ${CLASSICAL_LOW_TEMP}, \
		$(foreach optimizationCycles, ${OPTIMIZATIONCYCLES}, \
		$(foreach number, ${NUMBERS}, \
		results_classical_parameter_sweep/info_${filename}_${hightemp}_${lowtemp}_${optimizationCycles}_${number})))))

SIQUAN_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach temp, ${SIQUAN_TEMP}, \
		$(foreach transverse, ${TRANSVERSE_HIGH}, \
		$(foreach optimizationCycles, ${OPTIMIZATIONCYCLES}, \
		$(foreach lineRepresentation, ${LINEREPRESENTATION}, \
		$(foreach maxOrder, ${MAXORDER}, \
		$(foreach number, ${NUMBERS}, \
		results_sqa_sweep/info_${filename}_${temp}_${transverse}_${optimizationCycles}_${lineRepresentation}_${maxOrder}_${number})))))))

QUANTUM_ANNEALING_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach anneal_time, ${ANNEAL_TIME}, \
		$(foreach num_reads, ${NUM_READS}, \
		$(foreach slackvarfactor, ${SLACKVARFACTOR}, \
		$(foreach lineRepresentation, ${LINEREPRESENTATION}, \
		$(foreach maxOrder, ${MAXORDER}, \
		$(foreach chain_strength, ${CHAINSTRENGTH}, \
		$(foreach number, ${NUMBERS}, \
		results_qpu_sweep/info_${filename}_${anneal_time}_${num_reads}_${slackvarfactor}_${lineRepresentation}_${maxOrder}_${chain_strength}_${number}))))))))

QUANTUM_ANNEALING_READ_RESULTS = $(foreach filename, $(SWEEPFILES), \
		$(foreach anneal_time, ${ANNEAL_TIME}, \
		$(foreach num_reads, ${NUM_READS}, \
		$(foreach slackvarfactor, ${SLACKVARFACTOR}, \
		$(foreach lineRepresentation, ${LINEREPRESENTATION}, \
		$(foreach maxOrder, ${MAXORDER}, \
		$(foreach chain_strength, ${CHAINSTRENGTH}, \
		$(foreach sampleCutSize, ${SAMPLECUTSIZE}, \
		$(foreach number, ${NUMBERS}, \
		results_qpu_read_sweep/info_${filename}_${anneal_time}_${num_reads}_${slackvarfactor}_${lineRepresentation}_${maxOrder}_${chain_strength}_${sampleCutSize}_${number})))))))))

GLPK_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach timeout, $(TIMEOUT), \
		$(foreach number, ${NUMBERS}, \
		results_pypsa_glpk_sweep/info_${filename}_${timeout}_${number})))
		

## creating rules for result files


# define classical parameter sweep targets

define classicalParameterSweep
results_classical_parameter_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
	--env temperatureSchedule=[$(strip $(2)),$(strip $(3))] \
	--env transverseFieldSchedule=[0] \
	--env optimizationCycles=$(strip $(4)) \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 classical
	mkdir -p results_classical_parameter_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5)) results_classical_parameter_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach hightemp, ${CLASSICAL_HIGH_TEMP}, \
	$(foreach lowtemp, ${CLASSICAL_LOW_TEMP}, \
	$(foreach optimizationCycles, ${OPTIMIZATIONCYCLES}, \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call classicalParameterSweep, ${filename}, ${hightemp}, ${lowtemp}, ${optimizationCycles}, ${number})))))))


# define siquan sweep targets

define sqaParameterSweep
results_sqa_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
	--env temperatureSchedule=[$(strip $(2)),iF,0.0001] \
	--env transverseFieldSchedule=[$(strip $(3)),0.0] \
	--env optimizationCycles=$(strip $(4)) \
	--env lineRepresentation=$(strip $(5)) \
	--env maxOrder=$(strip $(6)) \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 sqa
	mkdir -p results_sqa_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7)) results_sqa_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach temp, ${SIQUAN_TEMP}, \
	$(foreach transverseField, ${TRANSVERSE_HIGH}, \
	$(foreach optimizationCycles, ${OPTIMIZATIONCYCLES}, \
	$(foreach lineRepresentation, ${LINEREPRESENTATION}, \
	$(foreach maxOrder, ${MAXORDER}, \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call sqaParameterSweep, ${filename}, ${temp}, ${transverseField}, ${optimizationCycles}, ${lineRepresentation}, ${maxOrder}, ${number})))))))))


# define D-Wave qpu targets

define qpuParameterSweep
results_qpu_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
	--env annealing_time=$(strip $(2)) \
	--env num_reads=$(strip $(3)) \
	--env slackVarFactor=$(strip $(4)) \
	--env lineRepresentation=$(strip $(5))\
	--env maxOrder=$(strip $(6)) \
	--env chain_strength=$(strip $(7)) \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8)) \
	--env inputNetwork=$(strip $(1)) \
	--env dwaveAPIToken=$(dwaveAPIToken) \
	energy:1.0 dwave-qpu
	mkdir -p results_qpu_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8)) results_qpu_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach anneal_time, ${ANNEAL_TIME}, \
	$(foreach num_reads, ${NUM_READS}, \
	$(foreach slackvarfactor, ${SLACKVARFACTOR}, \
	$(foreach lineRepresentation, ${LINEREPRESENTATION}, \
	$(foreach maxOrder, ${MAXORDER}, \
	$(foreach chain_strength, ${CHAINSTRENGTH}, \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call qpuParameterSweep, ${filename}, ${anneal_time}, ${num_reads}, ${slackvarfactor}, ${lineRepresentation}, ${maxOrder}, ${chain_strength}, ${number}))))))))))


# define targets for old sample data

define qpuReadSweep
results_qpu_read_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8))_$(strip $(9)): docker.tmp \
		results_qpu_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(9))
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
			--mount type=bind,source=$(PROBLEMDIRECTORY)/results_qpu_sweep/,target=/energy/results_qpu \
	--env annealing_time=$(strip $(2)) \
	--env num_reads=$(strip $(3)) \
	--env slackVarFactor=$(strip $(4)) \
	--env lineRepresentation=$(strip $(5))\
	--env maxOrder=$(strip $(6)) \
	--env chain_strength=$(strip $(7)) \
	--env sampleCutSize=$(strip $(8)) \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8))_$(strip $(9)) \
	--env inputNetwork=$(strip $(1)) \
	--env inputInfo=results_qpu/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(9)) \
	energy:1.0 dwave-read-qpu
	mkdir -p results_qpu_read_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6))_$(strip $(7))_$(strip $(8))_$(strip $(9)) results_qpu_read_sweep/

endef

$(foreach filename, $(SWEEPFILES), $(foreach anneal_time, ${ANNEAL_TIME}, \
	$(foreach num_reads, ${NUM_READS}, \
	$(foreach slackvarfactor, ${SLACKVARFACTOR}, \
	$(foreach lineRepresentation, ${LINEREPRESENTATION}, \
	$(foreach maxOrder, ${MAXORDER}, \
	$(foreach chain_strength, ${CHAINSTRENGTH}, \
	$(foreach sampleCutSize, ${SAMPLECUTSIZE}, \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call qpuReadSweep, ${filename}, ${anneal_time}, ${num_reads}, ${slackvarfactor}, ${lineRepresentation}, ${maxOrder}, ${chain_strength}, ${sampleCutSize}, ${number})))))))))))


# define glpk targets

define pypsa-glpk
results_pypsa_glpk_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3)) \
	--env inputNetwork=$(strip $(1)) \
	--env timeout=$(strip $(2)) \
	energy:1.0 pypsa-glpk
	mkdir -p results_pypsa_glpk_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3)) results_pypsa_glpk_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(foreach timeout, $(TIMEOUT), \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call pypsa-glpk, ${filename}, ${timeout}, ${number})))))

# end of creating rules for results



# Define further helper targets

docker.tmp: Dockerfile DockerInput/run.py DockerInput/Backends/SqaBackends.py DockerInput/Backends/IsingPypsaInterface.py DockerInput/Backends/PypsaBackends.py DockerInput/Backends/DwaveBackends.py
	$(DOCKERCOMMAND) build -t energy:1.0 . #&& touch docker.tmp


# all plots are generated using the python plot_results script
plots: venv/bin/activate
	mkdir -p plots && source venv/bin/activate && python plot_results.py

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


clean:
	rm -rf Problemset/info*
