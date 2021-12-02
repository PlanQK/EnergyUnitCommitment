SHELL := /bin/bash
DOCKERCOMMAND := docker
REPETITIONS := 1

PROBLEMDIRECTORY := $(shell git rev-parse --show-toplevel)
# alternative in case this is not a git repository
#PROBLEMDIRECTORY := $(shell pwd)

#for qpu sweeps, export an APIToken as:
#dwaveAPIToken

NUMBERS = $(shell seq 1 ${REPETITIONS})

SWEEPS = 10 30 77 215 599 1668 4641 12915 35938 100000
CLASSICAL_HIGH_TEMP = $(shell seq 10.0 10.0 10)
CLASSICAL_LOW_TEMP = $(shell seq 0.5 0.5 0.5)
SIQUAN_TEMP = $(shell seq 1.0 1 4)
TRANSVERSE_HIGH = $(shell seq 1.0 1.0 1)


ANNEAL_TIME = $(shell seq 100 50 100)
NUM_READS = $(shell seq 100 50 100)
SLACKVARFACTOR = $(shell seq 70 30 70)
GRANULARITY = $(shell seq 1 1 1)


INPUTFILES = $(shell find $(PROBLEMDIRECTORY)/inputNetworks -name "input*.nc" | sed 's!.*/!!' | sed 's!.po!!')
SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "input*.nc" | sed 's!.*/!!' | sed 's!.po!!')

CLASSICAL_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), $(foreach hightemp, ${CLASSICAL_HIGH_TEMP}, \
		$(foreach number, ${NUMBERS}, $(foreach lowtemp, ${CLASSICAL_LOW_TEMP}, \
		results_classical_parameter_sweep/info_${filename}_${hightemp}_${lowtemp}_${number}))))

SIQUAN_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), $(foreach temp, ${SIQUAN_TEMP}, \
		$(foreach transverse, ${TRANSVERSE_HIGH}, $(foreach number, ${NUMBERS}, results_sqa_sweep/info_${filename}_${temp}_${transverse}_${number}))))

QUANTUM_ANNEALING_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach anneal_time, ${ANNEAL_TIME}, \
		$(foreach num_reads, ${NUM_READS}, \
		$(foreach slackvarfactor, ${SLACKVARFACTOR}, \
		$(foreach granularity, ${GRANULARITY}, \
		$(foreach number, ${NUMBERS}, \
		results_qpu_sweep/info_${filename}_${anneal_time}_${num_reads}_${slackvarfactor}_${granularity}_${number}))))))

NETWORK_FILES = $(foreach filename, $(SWEEPFILES), \
		results_pypsa_glpk_sweep/info_${filename})


SQA_FILES := $(foreach filename, $(INPUTFILES), $(foreach sweep, ${SWEEPS}, \
	$(foreach number, ${NUMBERS}, results_sqa/info_${filename}_${sweep}_${number})))

CLASSICAL_FILES := $(foreach filename, $(INPUTFILES), $(foreach sweep, ${SWEEPS}, \
	$(foreach number, ${NUMBERS}, results_classical/info_${filename}_${sweep}_${number})))

#
# Define classical parameter sweep targets
#

define classicalParameterSweep
results_classical_parameter_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
	--env temperatureSchedule=[$(strip $(2)),$(strip $(3))] \
	--env transverseFieldSchedule=[0] \
	--env optimizationCycles=1000 \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 classical
	mkdir -p results_classical_parameter_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)) results_classical_parameter_sweep/

endef

$(foreach filename, $(SWEEPFILES), $(foreach hightemp, ${CLASSICAL_HIGH_TEMP}, \
	$(foreach lowtemp, ${CLASSICAL_LOW_TEMP}, $(foreach number, ${NUMBERS}, $(eval $(call classicalParameterSweep, ${filename}, ${hightemp}, ${lowtemp}, ${number}))))))

#
# Define siquan sweep targets
#

define sqaParameterSweep
results_sqa_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
	--env temperatureSchedule=[$(strip $(2))] \
	--env transverseFieldSchedule=[$(strip $(2)),0.0] \
	--env optimizationCycles=1000 \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 sqa
	mkdir -p results_sqa_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)) results_sqa_sweep/

endef

$(foreach filename, $(SWEEPFILES), $(foreach temp, ${SIQUAN_TEMP}, \
	$(foreach transverseField, ${TRANSVERSE_HIGH}, $(foreach number, ${NUMBERS}, $(eval $(call sqaParameterSweep, ${filename}, ${temp}, ${transverseField}, ${number}))))))

#
# Define D-Wave qpu targets
#

define qpuParameterSweep
results_qpu_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
	--env annealing_time=$(strip $(2)) \
	--env num_reads=$(strip $(3)) \
	--env slackVarFactor=$(strip $(4)) \
	--env granularity=$(strip $(5))\
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6)) \
	--env inputNetwork=$(strip $(1)) \
	--env dwaveAPIToken=$(dwaveAPIToken) \
	energy:1.0 dwave-qpu
	mkdir -p results_qpu_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4))_$(strip $(5))_$(strip $(6)) results_qpu_sweep/

endef

$(foreach filename, $(SWEEPFILES), $(foreach anneal_time, ${ANNEAL_TIME}, \
	$(foreach num_reads, ${NUM_READS}, \
	$(foreach slackvarfactor, ${SLACKVARFACTOR}, \
	$(foreach granularity, ${GRANULARITY}, \
	$(foreach number, ${NUMBERS}, \
	$(eval $(call qpuParameterSweep, ${filename}, ${anneal_time}, ${num_reads}, ${slackvarfactor}, ${granularity}, ${number}))))))))


define pypsa-glpk
results_pypsa_glpk_sweep/info_$(strip $(1)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
	--env outputInfo=info_$(strip $(1)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 pypsa-glpk
	mkdir -p results_pypsa_glpk_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1)) results_pypsa_glpk_sweep/

endef

$(foreach filename, $(SWEEPFILES), \
	$(eval $(call pypsa-glpk, ${filename})))

####################
### not functional yet
####################

#
# Define siquan simulation targets
#

define sqaRule
results_sqa/info_$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/inputNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/inputNetworks/,target=/energy/Problemset \
	--env temperatureSchedule=[TODO] \
	--env transverseFieldSchedule=[10, 0] \
	--env optimizationCycles=$(strip $(2)) \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 sqa
	mkdir -p results_sqa
	mv $(PROBLEMDIRECTORY)/inputNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3)) results_sqa

endef

#$(foreach filename, $(INPUTFILES), $(foreach sweep, ${SWEEPS}, \
	$(foreach number, ${NUMBERS}, $(eval $(call sqaRule, ${filename}, ${sweep}, ${number})))))

#
# Define classical simulation targets
#

define classicalRule
results_classical/info_$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/inputNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/inputNetworks/,target=/energy/Problemset \
	--env temperatureSchedule=[110,iF,0.4] \
	--env transverseFieldSchedule=[0] \
	--env optimizationCycles=$(strip $(2)) \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 classical
	mkdir -p results_classical
	mv $(PROBLEMDIRECTORY)/inputNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3)) results_classical

endef

#$(foreach filename, $(INPUTFILES), $(foreach sweep, ${SWEEPS}, \
	$(foreach number, ${NUMBERS}, $(eval $(call classicalRule, ${filename}, ${sweep}, ${number})))))

####################
### end 
####################


#
# Define further helper targets
#
docker.tmp: Dockerfile DockerInput/run.py DockerInput/Backends/SqaBackends.py DockerInput/Backends/IsingPypsaInterface.py DockerInput/Backends/PypsaBackends.py DockerInput/Backends/DwaveBackends.py
	$(DOCKERCOMMAND) build -t energy:1.0 . && touch docker.tmp


# all plots are generated using the python plot_results script
plots: venv/bin/activate
	mkdir -p plots && source venv/bin/activate && python plot_results.py

venv/bin/activate:
	python3 -m venv venv && pip install -r requirements.txt && pip install 

.PHONY: all clean classicalParSweep siquanParSweep quantumAnnealParSweep plots pypsa-glpk

all: classicalParSweep siquanParSweep quantumAnnealParSweep pypsa-glpk

classicalParSweep: $(CLASSICAL_PARAMETER_SWEEP_FILES)

siquanParSweep: $(SIQUAN_PARAMETER_SWEEP_FILES)

quantumAnnealParSweep: $(QUANTUM_ANNEALING_SWEEP_FILES)

pypsa-glpk: $(NETWORK_FILES)


classicalSimulation: $(CLASSICAL_FILES)

siquanSimulation: $(SQA_FILES) 

clean:
	rm -rf Problemset/info*
