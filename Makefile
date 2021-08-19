SHELL := /bin/bash
DOCKERCOMMAND := docker
OPTIMIZER := classical
REPETITIONS := 2
PROBLEMDIRECTORY := /mnt/c/Users/d60098/Documents/PlanQK/energy/DockerUseCase/inputNetworks

NUMBERS = $(shell seq 1 ${REPETITIONS})
SWEEPS = 10 77 215 599 1668 4641 12915 35938 100000

INPUTFILES = $(shell find $(PROBLEMDIRECTORY)/ -name "input*.nc" | sed 's!.*/!!' | sed 's!.po!!')

SIMULATION_FILES := $(foreach filename, $(INPUTFILES), $(foreach sweep, ${SWEEPS}, \
	$(foreach number, ${NUMBERS}, results_$(OPTIMIZER)/info_${filename}_${sweep}_${number})))



define simulationRule
results_$(OPTIMIZER)/info_$(strip $(1))_$(strip $(2))_$(strip $(3)): $(PROBLEMDIRECTORY)/$(strip $(1)) docker.tmp
	$(DOCKERCOMMAND) run --mount type=bind,source=$(PROBLEMDIRECTORY),target=/energy/Problemset \
	--env temperatureSchedule=[100,iF,0.0001] \
	--env transverseFieldSchedule=[0] \
	--env optimizationCycles=$(strip $(2)) \
	--env outputInfo=info_$(strip $(1))_$(strip $(2))_$(strip $(3)) \
	--env inputNetwork=$(strip $(1)) \
	energy:1.0 $(OPTIMIZER)
	mkdir -p results_$(OPTIMIZER)
	mv $(PROBLEMDIRECTORY)/info_$(strip $(1))_$(strip $(2))_$(strip $(3)) results_$(OPTIMIZER)

endef

$(foreach filename, $(INPUTFILES), $(foreach sweep, ${SWEEPS}, \
	$(foreach number, ${NUMBERS}, $(eval $(call simulationRule, ${filename}, ${sweep}, ${number})))))

docker.tmp: Dockerfile Dockerinput/run.py Dockerinput/Backends/SqaBackends.py Dockerinput/Backends/IsingPypsaInterface.py Dockerinput/Backends/PypsaBackends.py Dockerinput/Backends/DwaveBackends.py
	$(DOCKERCOMMAND) build -t energy:1.0 . && touch docker.tmp

# all plots are generated using the python plot_results script
plots/CostVsTraining.pdf: plot_results.py venv/bin/activate $(SIMULATION_FILES)
	mkdir -p plots && source venv/bin/activate && python plot_results.py

venv/bin/activate:
	python3 -m venev venv && pip install -r requirements.txt && pip install 


.PHONY: all
all: $(SIMULATION_FILES)

clean:
	rm -rf Problemset/info*