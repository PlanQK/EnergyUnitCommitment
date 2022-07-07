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

The JSON-object contains these four fields: `backend`, `backend_config`, `ising_interface` and `API_token`.
They specify which solver is going to be used and it's configuration.
The field `backend` contains a string denoting which solver to use and `backend_config` contains configuration parameters of that solver.
Most of the solvers require the unit commitment problem to be cast as a [quadratic unconstrained binary optimization problem (QUBO)](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization).
If a solver requires a QUBO formulation of the unit commitment problem, the field `ising_interface` specifies all relevant details on how the QUBO is built.
At last, the field `API_token` contains a JSON-object that can be used to pass API tokens of services which are used by solvers that access quantum hardware.


## Service Configuration
This service supports various solvers for obtaining solutions of the unit commitment problem. 
Currently, the solvers that can be used consist of:
- Mixed Integer Linear Programming (GLPK)
- Simulated Quantum Annealing
- Quantum Annealing
- Tabu search
- d-wave's hybrid QUBO solver

In principle, this service also contains a solver using the quantum approximation optimization algorithm, but it is currently unavailable due to a bug.
We also list simulated annealing, which is implemented as simulated quantum annealing with a vanishing transverse field.


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

In order to specifiy which QUBO an underlying solver uses as a basis to solve the unit commitment problem, you have to specify which methods are used to represent network components as QUBO-variables.
For each constraint and the optimization goal, you then have to specify which method is used to construct a QUBO using the aforementioned representation of network components as QUBO-variables.
Since the total QUBO is obtained by summing all QUBO's encoding a the optimization of some subproblem, those can be done independently.

As a rough outline of the underlying model, a QUBO-variable represents a quantized piece of energy with it's value denoting if that energy is available or vanishes.
A network component is then represented using groups of QUBO-variables with it's state being described by the various values these QUBO-variables can be assigned to.
The sum of the quantized energy represents the energy that that network component has in a particular state.

For each aspect of the the unit commitment, the corresponding value of that QUBO-variable with regard to that aspect can be calculated using the underlying value of the energy.
For example, if you want to handle the marginal costs, the value of the QUBO-variable belonging to a group that represents a generator, can be calculated as the product of the 
energy it represents and the margina costs this generator incurs for each unit of energy.


#### QUBO representation

In order to represent transmission lines, the capacity of the line has to be split up into multiple QUBO-variables.
We have decided to also impart a direction of the flow onto the QUBO-variables instead of encoding the direction of the flow into an addtional QUBO-variable.
This matters for the kirchhoff constraint because the direction of flow in a transmission line determines the sign of the energy at a bus

In order to specify how a line is represent as QUBO variables, we pass a method that takes the capacity of the line and returns a list of weights which represent pieces of energy.
This list of weights has to fulfill the condition, that no sum of entries of that list exceeds the capacity of the transmission line.
The weights also have to have signs which represent the direction of flow. 
The chosen list of weights determines how finely you can represent a flow of energy.
Thus, the sum of all positive weights, as well as the sum of all negative weights, should come close to the (negative) capacity of the transmission line
to represent the maximum possible flow.

The following table shows how to configure using different transmission line representation. The key for this is `formulation` in the the `ising_interface` field


|  parameter value |   description of line representation                                                                  | additional information
| ---------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------
| fullsplit        | Each QUBO-variable carries a unit of energy for each direction                                        | This scales badly with line capacity. Has a high runtime, but also higher quality of solution
| customsplit      | a compromise of minimizing the number of variables and minimizing the difference in magnitude         | only accepts integer values between 1 and 4 so far.
| cutpowersoftwo   | uses a binary representation in each direction with a cut off at the highest order                    | cheapest option in QUBO-variables wise, but has huge differences in magnitude of weights

As of now, generators are encoded with a single QUBO-variable.
Thus, they can only have two states: Committed and producing full power or turned off.
We will later add being able to represent a range of power, as well as specifying a minimal power output if they are committed

In total, this adds the following entries to the configuration.
```
{
    "ising_interface": {
        "formulation": {parameter value}
    }
}
```

### Encoding constraints

Since a QUBO is by definition unconstrained, we will not distinguish between constraints of the problem like the kirchhoff constraint or the optimimization goal.
When using a QUBO-based solver, various constraints can be represented as a QUBO by using a langrangian function whose minima are exactly those states, that satisfy the constraint.
The total QUBO is then simply the weighted sums of all langrangian function if they are all polynomial functions of order two.
Therefore, each other field in the `ising_interface` value contains the full information needed to construct the QUBO corresponding to the constraint indicated by the key.

The following table describes which key is used for which constraint of the problem.
Each of these options support the option `scale_factor` which is the factor used when summing up all QUBO's to obtain the full QUBO.

|  keyword             |  problem constraint description
| -------------------- | ----------------------------------------------------
| kirchhoff            | The kirchhoff constraint requires that the supplied energy is equal to the load at all buses
| marginal_cost        | The marginal costs is to be minimal among all feasible solutions

Any other key that isn't used to desribe how network components are encoded or in the above table as a listed constraint will be ignored.
Adding or removing any constraint from QUBO can simply be done by ommitting the key that corresponds to that constraint.
We will no go over the various constraints, and how we can choose different approach to encode it into a QUBO.

#### Kirchhoff constraint

The kirchhoff constraint enforces that a for a feasible solution, the load is equal to the supplied energy at each bus.
We model this as a QUBO by contructing a QUBO that enforces it at each bus.
That QUBO's interactions are setup in a way, that the cost of it is equal to the squared distance of the load at the bus to the supplied power.
Because this constraint is integral to the problem, it will always be added to the problem and only has the parameter `scale_factor`. 

For example, adding the kirchhoff constraint with a factor of `2.0` to the QUBO would require adding the following entry to the `ising_interface` :
```
{
    "ising_interface": {
        ""kirchhoff: 2.0
    }
}
```

This constraint also doesn't support any other encoding than the squared distance approach.

#### Marginal Costs

Minimizing the marginal costs is technically not a constraint that a solution has to satisfy, but just the optimization goal. 
Because we have to write it as QUBO, it is similar to a constraint and can be treated as such when configuring the solver.

Directly encoding the cost into the energy cost requires huge scaling factors, which in return leads to a bad spectral gap.
Therefore, most strategies that we implemented work by estimating the total marginal cost and encoding the squared distance to that estimation.
Using an estimation based approach has two benefits.
First, the quadratic growth of the squared distance ensures that the constraint will always outscale any linear factors.
This prevents the marginal costs to be irrelevant when used in conjuction with a small scaling factor.
The other benefit is that good estimations lead to well conditioned problems since it means that both the kirchhoff constraint and the marginal costs
can be optimally solved by the same solution.
The following table describes various strategies to encode the marginal cost into a QUBO. 


|    configuration keyword            |                 strategy description
| ----------------------------------- | ----------------------------------------------------------------------  
| marginal_as_penalty                 | direct translation from cost to energy as first order interactions                                  
| global_cost_square                  | squared distance of total cost to a fixed estimation                    
| global_cost_square_with_slack       | squared distance of total cost to an estimation with slack variables
| local_marginal_estimation_distance  | squared distance of total cost at each bus to estimations               

Each strategy supports the keys `scale_factor` and `offset_estimation_factor` whose values are floats. 
The former simply scales the QUBO encoding the marginal cost by a linear factor while the latter introduces an offset into the marginal cost values of the generators based on their relative power output. 
It does so by using a reference value and the `offset_estimation_factor` to calculate the total offset of a solution that satisfies the kirchhoff constraint.
Using the latter constraint, that offset can be distributed across all generators, using the constraint to guarantee that this change in marginal costs doesn't alter the optimal solution of the problem.

A factor of `0.0` introduces no offset of the marginal costs. 
The factor `1.0` introduces an offset equal to the reference value.
That value is calculated by commiting the most efficient generators regardless of their associated bus up to the total load of the problem.
In essence, this reference value is the most simple lower bound of the cost of the optimal solution. 

In order to reduce differences in magnitudes of different interactions, we have slightly changed how the distance to the estimation is encoded.
Instead of adding a parameter that is used to formulate a QUBO that describes the squared distance to that value, we instead assume that the marginal costs of the optimal solution are always zero.
We adjust the estimation of the marginal costs by using various offsets.
The goal is to find an offset of the marginal costs, that the optimal solution has neglibile distance to zero with regards to the offset marginal costs.
The difference to just encoding the squared distance to some value is that that leads to huge numbers on the diagonal of the corresponding hamiltonian matrix.
Our approach allows us to transform first-order interactions into second-order interactions, using the kirchhoff constraint to guarantee that the combined problem
have the same solutions.


The option global_cost_square_with_slack has three more options on top of the `offset_estimation_factor`. 
The idea is to introduce slack variables that act like generators, that are irrelevant to the kirchhoff constraint, but, if active,
contribute some fixed value to the marginal cost. Thus it can be used to slightly adjust the estimation during the optimization run. The following table describes the parameters that are used
to describe these slack variables.

|  keyword slack variables |  description 
| ------------------------ | ------------------------------------------------------ 
|  slack_type              | a string that specifies by which rule the slack weights are generated. 
|  slack_scale             | additional linear scaling value for the slack variable weights
|  slack_size              | number of slack variables to be generated according to the slack_type rule

The only supported value `slack_type` is `binary_power` which configures weights of slack variables as ascending powers of two. 
In conjuction with `slack_size`, which specifies how many slack variables are created, the slack marginal cost can be set up as as any fixed length binary number.
The `slack_scale` works similar to `scale_factor`, being a float that scales all slack weights, allowing either bigger or smaller than integer step size.

In total, the JSON-object for configuring the QUBO looks like this. 
The following JSON-can be used an a template, using the above tables to swap out the appropiate values.

```
{
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
}
```


### Backend Config

At last we describe the different solvers that this service provides, how to use them and which options you have for configuring them.
Each solver that you can choose solves a QUBO, so the difference between them is how they solve it.
The only exception are the solvers that solve a mixed integer linear program like glpk.
We now go through the different solvers and describe how you can use `backend_config` to configure it.

The default QUBO-based solver is simulated quantum annealing.

#### Simulated Quantum Annealing
This solver solves a QUBO using a monte carlo simulation of quantum annealing to find an optimal solution. 
You can find more information on simulated quantum annealing [here](https://platform.planqk.de/algorithms/4ab6ed1f-9f5e-4caf-b0b2-59d1444340d1/) and on quantum annealing [here](https://platform.planqk.de/algorithms/786e1ff5-991e-428d-a538-b8b99bc3d175/). More on the used solver itself can be found [here](https://platform.planqk.de/services/ff5c0cdd-4a87-4473-8086-cb658a9f85a2)
In this particular case, this service uses the same implementation as the SQA service, but also provides the tools to construct the QUBO, as well as interpret a solution as a state of the network.

The following table shows which  parameters can be adjusted.

Parameter Name |  Default Value    |    Description                                                                                                |  Runtime Impact            
-------------  | ----------------- |   ----------------------------------------------------------------------------------------------------------  |  -----------------  
H              | "[8.0,0.0]"       |  The transverse field strength schedule. It describes the interaction strength between trotter slices         |  None
T              | "[0.1,iF,0.0001]" |  The temperature schedule. At higher temperatures, qubit states are more likely to change                     |  None
trotter        | 32                |  The number of trotter slices. They are an approximation of the quantum state of quantum annealing            |  grows linearly
steps          | 20                |  The number of discrete time steps used to simulate the continuous evolution of the quantum state             |  grows linearly
seed           | N/A               |  The seed used by the random number generator for the solver. Unless explicitly set, the seed is random       |  None
time_limit     | None              |  An upper limit of the estimated runtime. The service will fail right away if this limit is exceeded          |  None or stops the service

You can see that the first two parameters both describe a schedule. Thus, we have to explain what a schedule is and how it affects the solver.
Simulated quantum annealing simulates the continous evolution of a quantum state in multiple discrete time steps.
The two schedules determine the strength of two aspects of the simulation at each time step.
Similar to classical annealing, the temperature is a measure of how much change can happen in one time step, while the transverse field is a measure of
interaction strength between different trotter slices.

##### Specifying a Schedule

A schedule is a function $f \colon [0,1] \rightarrow \mathbb{R}$ of the unit interval. 
If the simulation consists of $n+1$ steps with the zeroth step $t_0 = 0$ and last step $t_n = 1$, the strength of a field according to that schedule at the $i$-th step is given as $f(\frac{i}{n})$. The string that describes the
schedule is simply a shorthand for refering to a member of a  family of functions that can be used as schedules. You can also concatenate different schedules in which case they get evenly 
distributed over the unit interval. The following table gives an overview over admissable strings.

Schedule String    |  Function                                                                                                        | Description
----------------   | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------
[10,1] or [10,l,1] |  $p \mapsto 10 - 9 \cdot p$                                                                                      | a linear ramp from 10 to 1 
[a,iF,b]           |  $p \mapsto \frac{a \cdot b}{b + (a - b) \cdot p}$                                                               | This is fast at the beginning and slow towards the end
[a,iS,b]           |  $p \mapsto a + b - \frac{a \cdot b}{a - (a - b) \cdot p}$                                                       | This is slow at the beginning and fast towards the end
[a,sS,b]           |  $p \mapsto a + (b - a) \cdot p^2$                                                                               | N/A
[a,sF,b]           |  $p \mapsto b + (a - b) \cdot (p-1)^2$                                                                           | N/A
[10,l,2,2,l,1]     |  $p \mapsto \begin{cases} 10 - 16 \cdot p & p \leq \frac{1}{2} \\ 3 - 2 \cdot p & p \geq \frac{1}{2} \end{cases}$| first a linear ramp from 10 to 2 and then in the same time a linear ramp from 2 to 1

Any string that adheres to the rules above is a valid value for the temperature or transverse field schedule.
As general advice, both the number of trotter slices, as well as time steps increase the solution quality, but also increase the runtime linearly.
Increasing them however has dmininishing returns on solution quality, with the number of time steps dminishing faster.

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
