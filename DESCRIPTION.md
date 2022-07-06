## Energy Unit Commitment
This service contains various solvers for optimizing a simplified version of the unit commitment problem of an energy grid. 
The unit commitment problem consists of an energy grid consisting of power generators, transmission lines and loads. 
The problem is to optimally choose which generators to commit such that all loads are met and the generators incur minimal costs. 
You can find more information on the problem [here](https://platform.planqk.de/use-cases/e8664221-933b-4410-9880-80a6900c9f86/).

### input format
The input of this service has to be in JSON-format. This input has the two fields `data` and `params` to specifiy the network and which solver to use respectively.
For the field `data`, which contains the network, this service requires it to be a serialized [PyPSA network](https://pypsa.readthedocs.io/en/latest/).
You can obtain a serialized PyPSA network by first exporting it to an xarray using [export_to_netcdf](https://pypsa.readthedocs.io/en/latest/api_reference.html#pypsa.Network.export_to_netcdf).
Calling the resulting xarray's method `to_dict()` returns a dictionary which can be saved in JSON-format.

In order to choose the solver and configure it's parameters, the field `params` is used to pass another JSON-object.
Because there are multiple solvers and each solvers has some unique parameters, there are a lot of possible options to specify in that field.
Therefore we will go through the structure of the JSON object and explain how you can configure the various aspects of the different solvers.

The configuration parameters JSON-object contains these four fields: `backend,  backend_config, ising_interface, API_token`.
They specify which solver is going to be used and it's configuration.
The field `backend` contains a string denoting which solver to use and `backend_config` contains configuration parameters of that solver.
If a solver requires a QUBO formulation of the unit commitment problem, the field `ising_interface` specifies all relevant details on how the QUBO is built.
At last, the field `API_token` contains a JSON-object that can be used to pass API tokens of services which offer access to quantum hardware.


## Service Configuration
This service supports various solvers for obtaining solutions of the unit commitment problem. 
Currently, the solvers that can be used consist of:
- Mixed Integer Linear Programming (GLPK)
- Simulated Quantum Annealing
- Quantum Annealing
- Tabu search

In principle, this service also contains a solver using the quantum approximation optimization algorithm, but it is currently unavailable due to a bug.


The following table lists all supported solvers and how to choose them by passing the correct string to the `backend` field. Each solver also supports an additional
configuration keyword which can also be used to passed parameters.
If both the `backend` and the solver specific configuration field set the same parameter, the latter value takes precedence.

|   solver           |   keyword    |   description                                                                               |  configuration keyword  |  uses QUBO  | API token
| ------------------ | ------------ | ------------------------------------------------------------------------------------------- | ----------------------- | ----------- | ---------
| sqa                | sqa          | performs simulated quantum annealing. Default solver if none is speficied                   |  sqa_backend            |  Yes        | None
| annealing          | classical    | performs (classical) simulated annealing                                                    |  sqa_backend            |  Yes        | None
| tabu search        | dwave-tabu   | performs tabu search as it is in d-waves ocean package                                      |  dwave_backend          |  Yes        | None
| hybrid solver      | dwave-hybrid | uses d-waves hybrid solver in the cloud                                                     |  dwave_backend          |  Yes        | d-wave
| quantum annealing  | dwave-qpu    | performs quantum annealing using d-waves quantum annealer                                   |  dwave_backend          |  Yes        | d-wave
| qaoa               | qaoa         | performs QAOA using IBM's qiskit by either simulating or accessing IBM's quantum computer   |  qaoa_backend           |  Yes        | IBMQ
| glpk               | pypsa-glpk   | solves a mixed integer linear program obtained by pypsa using the GLPK solver               |  pypsa_backend          |  No         | None


Keep in mind that classical simulated annealing is just simulated quantum annealing without a transverse field coupling multiple trotter slices. 
Since the main focus of this service lies on (simulated) quantum annealing and the MILP reference solution, configuring tabu search and the hybrid algorithm will be added later.
Thus, choosing a solver adds the following entries the `params` dictionary:

```
{
    "backend": "{keyword}",
    "{configuration keyword}": {}
}
```

Because all solvers except the mixed integer linear program require a QUBO formulation, we will explain how to configure the QUBO model first.

### Configuring the QUBO-model

In order to specifiy which QUBO an underlying solver uses as a basis to solve the unit commitment problem, you have to specify which methods are used to represent network components as qubits.
For each constraint and the optimization goal, you then have to specify which method is used to construct a QUBO using the aforementioned representation of network components as qubits.
Since the total QUBO is obtained by summing all QUBO's encoding a the optimization of some subproblem, those can be done independently.

As a rough outline of the underlying model, a QUBO-variable represents a quantized piece of energy with it's value denoting if that energy is available or vanishes.
A network component is then represented using groups of QUBO-variables with it's states being described by the various values these QUBO-variables can be assigned to.
The sum of the quantized energy represents the energy that that network component has in a particular state.

For each aspect of the the unit commitment, the corresponding value of that qubit with regard to that aspect can be calculated using the underlying value of the energy.
For example, if you want to handle the marginal costs, the value of the qubit belonging to a group that represents a generator, can be calculated as the product of the 
energy it represents and the margina costs this generator incurs for each unit of energy.


#### Qubit representation

In order to represent transmission lines, the capacity of the line has to be split up into multiple QUBO-variables.
We have decided to also impart a direction of the flow onto the qubits instead of encoding the direction of the flow into an addtional QUBO-variable.
This matters for the kirchhoff constraint because the direction of flow in a transmission line determines the sign of the energy at a bus

In order to specify how a line is represent as QUBO variables, we pass a method that takes the capacity of the line and returns a list of weights which represent pieces of energy.
This list of weights has to fulfill the condition, that no sum of entries of that list exceeds the capacity of the transmission line.
The weights also have to have signs which represent the direction of flow. 
The chosen list of weights determines how finely you can represent a flow of energy.
Thus, the sum of all positive weights, aswell as the sum of all negative weights, should come close to the (negative) capacity of the transmission line
to represent the maximum possible flow.

The following table shows how to configure using different transmission line representation. The key for this is `formulation` in the the `ising_interface` field


|  parameter value |   description of line representation                                                                  | additional information
| ---------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------
| fullsplit        | Each qubit carries $1$ unit of energy for each direction                                              | This scales badly with line capacity. Has a high runtime, but also higher quality of solution
| customsplit      | a compromise of minimizing the number of qubits and minimizing the difference in magnitude of weights | hard coded to split integer values between 1 and 4
| cutpowersoftwo   | uses a binary representation in each direction with a cut off at the highest order                    | cheapest option qubit wise, but has huge differences in magnitude of weights


When using the QUBO-based solver, various constraints can be encoded into the QUBO. Each constraint can be added by adding a dictionary to the json object using the corresponding key.
The following table describes which key to use for which aspect of the problem. Each of these options support the option `scale_factor` which is a float that linearly scales
the interactions of the QUBO encoding that option.

|   configuration key  |  aspect            
| -------------------- | ----------------------------------------------------
| formulation          | This specifies how transmission lines are encoded
| kirchhoff            | The kirchhoff constraint requires that the supplied energy is equal to the demand
| marginal_cost        | This adds the minimization of the marginal costs to the problem

#### Kirchhoff constraint
This models that at each bus the supplied energy equals the used energy. This constraint will always be added to the problem and only has the parameter `scale_factor`. From a technical perspective
the encoding is archieved by constructing a QUBO at eac bus that encodes the squared distance of the supplied power at that bus to the load at that bus.

#### Marginal Costs
The following table describes various strategies to encode the marginal cost into a QUBO. Directly encoding the cost into the energy cost requires huge scaling factors, which in return
leads to a bad spectral gap. Therefore, most strategies work by estimating the marginal cost and encoding the squared distance to that estimation.

|    strategy name                    |                 description                                             
| ----------------------------------- | ----------------------------------------------------------------------  
| marginal_as_penalty                 | direct translation from cost to energy as first order interactions                                  
| global_cost_square                  | squared distance of total cost to a fixed estimation                    
| global_cost_square_with_slack       | squared distance of total cost to an estimation with slack variables
| local_marginal_estimation_distance  | squared distance of total cost at each bus to estimations               

Each strategy supports the keys `scale_factor` and `offset_estimation_factor` whose values are floats. The former simply scales the QUBO encoding the marginal cost by a linear factor.
The other introduces an offset into the marginal cost values of the generators based on their relative power output. The offset is specified as a factor to some reference value. A factor
of `0.0` introduces no offset of the marginal costs. The factor `1.0` introduces an offset, such that the solution obtained by commiting the most efficient generators regardless of their associated bus up
to the total load of the problem would incurrs no cost. In essence, this reference value is a simple lower bound of the cost of the optimal solution. This offset estimation factor can be used
to adjust the estimation of strategies that encode the distance because (including the offset) they estimate a marginal cost of zero. So if the reference value is some float `x`
and the estimation factor is a float `y`, then the estimation to which the suared distance is encoded, is given by `x * y`.

The option global_cost_square_with_slack has three more options. The idea is to introduce slack variables that act like generators, that are irrelevant to the kirchhoff constraint, but if active
contribute some fixed value to the marginal cost. Thus it can be used to slightly adjust the estimation during the optimization run. The following table describes the parameters that are used
to describe these slack variables.

|  parameter          |  description 
| ------------------- | ------------------------------------------------------ 
|  slack_type         | a string that specifies by which rule the slack weights are generated. 
|  slack_scale        | additional linear scaling value for the slack variable weights
|  slack_size         | number of slack variables to be generated according to the slack_type rule


In total, the entry for configuring the QUBO looks like this. Unused parameters are ignored.
```
"ising_interface": {
    "formulation": "option of the transmission line representation",
    "kirchhoff": {
        "scale_factor": 1.0
    },
    "marginal_cost": {
        "formulation": "name of the theoretical model",
        "scale_factor": 0.3,
        "offset_estimation_factor": 1.0
        "slack_type": "binary_power",
        "slack_scale": 0.1,
        "slack_size": 3,
    }
}
```


### backend_config

At last we describe the different solvers and their respective parameters. While these parameters can be supplied in the field `backend_config`, each type of solver
also supports a secondary field from which to take parameter values.


#### Simulated Quantum Annealing
This solver casts the problem as a [quadratic unconstrained binary optimization problem](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) (QUBO) and solves it uses monte carlo simulation of quantum annealing 
to find an optimal solution. You can find more information on simulated quantum annealing [here](https://platform.planqk.de/algorithms/4ab6ed1f-9f5e-4caf-b0b2-59d1444340d1/) and on quantum annealing [here](https://platform.planqk.de/algorithms/786e1ff5-991e-428d-a538-b8b99bc3d175/). More on the used solver itself can be found [here](https://platform.planqk.de/services/ff5c0cdd-4a87-4473-8086-cb658a9f85a2)

#### Quantum Annealing
Quantum Annealing uses D-Wave's quantum hardware to perform quantum annealing. For that, it uses the advantage 4.1. This requires an API - token. Due to to hardware limitations, problem size is significantly
limited and solutions aren't as accurate as other solvers. You can find more information on the quantum annealer [here](https://docs.ocean.dwavesys.com/en/stable/overview/qpu.html). The service builds the QUBO
and commits it to d-wave's cloud service. 

The most important parameters for this solvers are listed in the following table:
|  parameter       |  description  
| ---------------- | ------------------------------------
| annealing_time   | length of one annealing procedure in microseconds  
| num_reads        | number of repetitions of quantum annealing
| chain_strength   | the strength of chains used to embed the QUBO onto the working graph of the annealer
| timeout          | time that the embedding algorith has to embed the QUBO onto the working graph before it fails

Due to noise in the samples, this solver also performs simple flow optimization to improve the results. Both annealing time and number of reads
improve the solution. In general more annlealing runs are better than annealing linger. The chain strength has to adjusted to the problem at hand.
If the strength is two small, breaks in chains break the theroretical model of the problem which leads to bad solutions. If the chain strength
is to high, it's magnitude overshadows the optimization of the actual problem, once again, leading to worse results. In total, an example for the
configuration of the quantum annealer looks like this:

```
"dwave_backend": {
    timeout: 100
    annealing_time: 50
    num_reads: 200
    chain_strength: 70
}
```



#### Quantum Approximation Optimization Algorithm (QAOA)
The quantum approximation optimization algorithm build a parametrized quantum circuit to solve the problem. Then, it uses a classical optimizer to adjust the parameters so the
circuit better fits the problem. You can either simulate the circuit, or use access to IBM's quantum computer to run it. If using a quantum computer or the noise model of a quantum computer, this solver requires
an API token.

Right now, the QAOA solver is broken.


#### Mixed Integer Linear Programming
The [mixed integer linear programming approach](https://en.wikipedia.org/wiki/Linear_programming) uses pypsa's native method to cast the problem into a mixed integer linear program and solve it using a solver.
The service has GLPK installed but you can also return a pyomo model of the problem to run it locally. Since is just a wrapper for pypsa native solving method, this has just one option

|   configuration key  |  impact            
| -------------------- | ------------------ 
| timeout              | time in seconds after which the optimization will be stopped and the current best result will be returned


### API-token
Using D-Wave's quantum hardware or using an IBM quantum computer for qaoa requires an API token.
For d-wave services, you have to use the key `dwave_API_token`, and for ibm, the key is 
`IMBQ_API_token`.

```
"API_token": {
    "IBMQ_API_token": "Your IMBQ API token",
    "dwave_API_token": "Your d-wave API token"
}
```
