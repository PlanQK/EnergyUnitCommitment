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
# doesn't exist, it will be created. If no folder is specified, one will 
# be created using the solver name as 'results_SOLVER_sweep/'
# If specifiying your own folder, DON'T forget '/' for a valid folder name
SAVE_FOLDER := 

###### define config file ######
# this file is the default file which contains values for all valid configurations of all solvers.
# Making will generate a run for each config file specified here. Other config files can be saved
# in /src/Configs
CONFIGFILES = config-all.yaml
#CONFIGFILES = $(shell find $(PROBLEMDIRECTORY)/src/Configs -name "config_[9][4-4].yaml" | sed 's!.*/!!' | sed 's!.po!!')

###### define sweep files ######
# Choose a regex that will be used to search the sweepNetworks folder for networks. 
# The default network is a randomly generated network containing 10 buses with 
# generators that produce integer valued power and a total load of 100
NETWORKNAME = defaultnetwork.nc
# NETWORKNAME = testNetwork4QubitIsing_2_0_20.nc 
# NETWORKNAME = testNetwork5QubitIsing_2_0_20.nc 
# NETWORKNAME = 220124cost5input_[1]0_[0]_20.nc

# lists networks to be used using NETWORKNAME
SWEEPFILES = $(shell find $(PROBLEMDIRECTORY)/sweepNetworks -name "$(strip $(NETWORKNAME))" | sed 's!.*/!!' | sed 's!.po!!')

###### define extra parameter ######
# Please check the current config-all.yaml for a list and description of all
# possible options in src/Configs.
# The name of the parameter has to be stored in a variable with the
# PARAMETER_ prefix and the values for it in a variable with the
# VAL_PARAMETER_ prefix. Only Parameters which have a name with the PARAMETER_ prefix
# will be read and added to the config. You can enable/disable them by uncommenting them.
#
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

#PARAMETER_KIRCHSCALEFACTOR = \
	"IsingInterface__kirchhoff__scaleFactor"
VAL_PARAMETER_KIRCHSCALEFACTOR = "1.0__5.0__10.0"

PARAMETER_KIRCHFACTOR = \
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


### SQA Parameters
#PARAMETER_TRANSVERSEFIELD = \
	"SqaBackend__transverseFieldSchedule"
VAL_PARAMETER_TRANSVERSEFIELD = "8.0"

#PARAMETER_SIQUAN_TEMP = \
	"SqaBackend__temperatureSchedule"
VAL_PARAMETER_SIQUAN_TEMP = "0.1"

#PARAMETER_TROTTERSLICES = \
	"SqaBackend__trotterSlices"
VAL_PARAMETER_TROTTERSLICES = "10__20__30__40__50__60__70__80__90__100"

#PARAMETER_OPTIMIZATIONCYCLES = \
	"SqaBackend__optimizationCycles"
VAL_PARAMETER_OPTIMIZATIONCYCLES = "5__10__15__20"


### D-Wave Quantum Annealer Parameters.
# Requires an APIToken set in the config
#PARAMETER_ANNEAL_TIME = \
	"DWaveBackend__annealing_time"
VAL_PARAMETER_ANNEAL_TIME = "100"

#PARAMETER_NUM_READS = \
	"DWaveBackend__num_reads"
VAL_PARAMETER_NUM_READS = "200"

#PARAMETER_CHAINSTRENGTH = \
	"DWaveBackend__chain_strength"
VAL_PARAMETER_CHAINSTRENGTH = "60"

#PARAMETER_PROGTHERMALIZATION = \
	"DWaveBackend__programming_thermalization"
VAL_PARAMETER_PROGTHERMALIZATION = "1"

#PARAMETER_READTHERMALIZATION = \
	"DWaveBackend__readout_thermalization"
VAL_PARAMETER_READTHERMALIZATION = "1"

#PARAMETER_SAMPLECUTSIZE = \
	"DWaveBackend__sampleCutSize"
VAL_PARAMETER_SAMPLECUTSIZE = "200"

#PARAMETER_STRATEGY = \
	"DWaveBackend__strategy"
VAL_PARAMETER_STRATEGY = "LowestEnergy"

#PARAMETER_POSTPROCESS = \
	"DWaveBackend__postprocess"
VAL_PARAMETER_POSTPROCESS = "flow"

#PARAMETER_DWAVETIMEOUT = \
	"DWaveBackend__timeout"
VAL_PARAMETER_DWAVETIMEOUT = "100"

#PARAMETER_SAMPLEORIGIN = \
	"DWaveBackend__sampleOrigin"
VAL_PARAMETER_SAMPLEORIGIN = \
	"infoNocost_220124cost5input_10_0_20.nc_300_200_fullsplit_60_1"


### GLPK Parameter
#PARAMETER_PYPSASOLVERNAME = \
	"PypsaBackend__solver_name"
VAL_PARAMETER_PYPSASOLVERNAME = "glpk"

#PARAMETER_PYPSATIMEOUT = \
	"PypsaBackend__timeout"
VAL_PARAMETER_PYPSATIMEOUT = "60"

###### extra parameter string generation ######

# combine each parameter name with its values, if it is not commented out
EXTRAPARAMSEPARATE = 	$(foreach name, $(filter PARAMETER_%,$(.VARIABLES)), \
						$(foreach value, ${VAL_${name}}, \
						${${name}}___${value}))

# join all separate parameter___value pairs
EXTRAPARAM = 	$(subst " ","____",$(foreach param, \
				${EXTRAPARAMSEPARATE},$(param)))

# if no Parameters are declared in the Makefile, the string will be set to an
# empty string
ifeq ($(EXTRAPARAM),)
EXTRAPARAM = ""
endif


###### generate default SAVE_FOLDER path for current solver if no path is specified
ifeq ($(SAVE_FOLDER),)
SAVE_FOLDER = results_$(strip $(VAL_PARAMETER_BACKEND))_sweep/
endif

###### result files of computations ######

GENERAL_SWEEP_FILES = $(foreach filename, $(SWEEPFILES), \
		$(foreach config, ${CONFIGFILES}, \
		${SAVE_FOLDER}${filename}_${config}_${EXTRAPARAM}))


###### creating rules for result files ######
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

clean:
	rm -rf Problemset/info*

# create rules for make using a specific solver and setting the corresponding SAVE_FOLDER
SOLVERS = classical sqa qpu qpu_read pypsa_glpk qaoa
#TODO make rules
