SHELL := /bin/bash
DOCKERCOMMAND := docker
REPETITIONS := 5
PROBLEMDIRECTORY := /mnt/c/EnergyUseCase/EnergyUnitCommitment

NUMBERS = $(shell seq 1 ${REPETITIONS})

SWEEPS = 10 77 215 599 1668 4641 12915 35938 100000

CLASSICAL_HIGH_TEMP = $(shell seq 10.0 10.0 100)
CLASSICAL_LOW_TEMP = $(shell seq 0.5 0.5 10)
QUANTUM_TEMP = $(shell seq 0.01 0.5 8)
TRANSVERSE_HIGH = $(shell seq 1.0 1.0 10)


INPUTFILES = $(shell find $(PROBLEMDIRECTORY)/inputNetworks -name "input*.nc" | sed 's!.*/!!' | sed 's!.po!!')
SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "input*.nc" | sed 's!.*/!!' | sed 's!.po!!')

CLASSICAL_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), $(foreach hightemp, ${CLASSICAL_HIGH_TEMP}, \
		$(foreach number, ${NUMBERS}, $(foreach lowtemp, ${CLASSICAL_LOW_TEMP}, \
		results_classical_parameter_sweep/info_${filename}_${hightemp}_${lowtemp}_${number}))))

QUANTUM_PARAMETER_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), $(foreach temp, ${QUANTUM_TEMP}, \
		$(foreach transverse, ${TRANSVERSE_HIGH}, $(foreach number, ${NUMBERS}, results_sqa_sweep/info_${filename}_${temp}_${transverse}_${number}))))

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
	--env optimizationCycles=10000 \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 classical
	mkdir -p results_classical_parameter_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)) results_classical_parameter_sweep/

endef

#$(foreach filename, $(SWEEPFILES), $(foreach hightemp, ${CLASSICAL_HIGH_TEMP}, \
#	$(foreach lowtemp, ${CLASSICAL_LOW_TEMP}, $(foreach number, ${NUMBERS}, $(eval $(call classicalParameterSweep, ${filename}, ${hightemp}, ${lowtemp}, ${number}))))))

#
# Define quantum sweep tagets
#

define sqaParameterSweep
results_sqa_sweep/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)): $(PROBLEMDIRECTORY)/sweepNetworks/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY)/sweepNetworks/,target=/energy/Problemset \
	--env temperatureSchedule=[$(strip $(2))] \
	--env transverseFieldSchedule=[$(strip $(2)),0.0] \
	--env optimizationCycles=10000 \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 sqa
	mkdir -p results_sqa_sweep
	mv $(PROBLEMDIRECTORY)/sweepNetworks/info_$(strip $(1))_$(strip $(2))_$(strip $(3))_$(strip $(4)) results_sqa_sweep/

endef

$(foreach filename, $(SWEEPFILES), $(foreach temp, ${QUANTUM_TEMP}, \
	$(foreach transverseField, ${TRANSVERSE_HIGH}, $(foreach number, ${NUMBERS}, $(eval $(call sqaParameterSweep, ${filename}, ${temp}, ${transverseField}, ${number}))))))

#
# Define quantum simulation targets
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

#
# Define further helper targets
#
docker.tmp: Dockerfile Dockerinput/run.py Dockerinput/Backends/SqaBackends.py Dockerinput/Backends/IsingPypsaInterface.py Dockerinput/Backends/PypsaBackends.py Dockerinput/Backends/DwaveBackends.py
	$(DOCKERCOMMAND) build -t energy:1.0 . && touch docker.tmp

# all plots are generated using the python plot_results script
plots/CostVsTraining.pdf: plot_results.py venv/bin/activate $(SQA_FILES)
	mkdir -p plots && source venv/bin/activate && python plot_results.py

venv/bin/activate:
	python3 -m venev venv && pip install -r requirements.txt && pip install 


.PHONY: all clean simulations classicalParSweep quantumParSweep
all: classicalParSweep simulations clasicalSimulation quantumParSweep

classicalParSweep: $(CLASSICAL_PARAMETER_SWEEP_FILES)

classicalSimulation: $(CLASSICAL_FILES)

quantumParSweep: $(QUANTUM_PARAMETER_SWEEP_FILES)

quantumSimulation: $(SQA_FILES) 

clean:
	rm -rf Problemset/info*
