# Energy Unit Commitment
This service contains various solvers for optimizing a simplified version of the unit commitment problem of an energy grid. The unit commitment problem is made up of an energy grid consisting of power generators, transmission lines and loads over multiple time steps.  The problem at hand is to optimally choose which generators to commit such that all loads at each time step are met while incurring minimal costs. You can find more information on the problem [here](https://platform.planqk.de/use-cases/e8664221-933b-4410-9880-80a6900c9f86/). The unit commitment problem is only concerned with running the grid and not with expanding it and minimizing capital costs.

# Contents

- [Input format](#Input_format)

  - [Building the network in PyPSA](#Building_the_network_in_PyPSA)

  - [Converting a network to JSON](#Converting_a_network_to_JSON)

- [Service Configuration](#Service_Configuration)

    - [Supported Solvers](#Supported_Solvers)

    - [Configuring the QUBO-model](#Configuring_the_QUBO-model)

        - [QUBO representation of a transmission line](#QUBO_representation_of_a_transmission_line)

        - [QUBO representation of a generator](#QUBO_representation_of_a_generator)

    - [Encoding constraints](#Encoding_constraints)

      - [Kirchhoff constraint](#Kirchhoff_constraint)

      - [Marginal Costs](#Marginal_Costs)

    - [Solver Configuration](#Solver_Configuration)

        - [Simulated Quantum Annealing](#Simulated_Quantum_Annealing)

          - [Specifying a Schedule](#Specifying_a_Schedule)

        - [Quantum Annealing](#Quantum_Annealing)

        - [Quantum Approximation Optimization Algorithm (QAOA)](#Quantum_Approximation_Optimization_Algorithm_(QAOA))

        - [Mixed Integer Linear Programming](#Mixed_Integer_Linear_Programming)

        - [API tokens](#API-token)

- [Output format](#Output format)


# Input format

The input for this service is a JSON-object. This object has the two fields `data` and `params` to specifiy the problem and which solver to use respectively. 
The field `data` contains a serialized [PyPSA network](https://pypsa.readthedocs.io/en/latest/).
The other field `params` contains the name of the solver to be used and configuration parameters for that solver.

First we go through the steps of building a PyPSA network that we can then pass to the service.
The PyPSA documentation also contains an [example](https://pypsa.readthedocs.io/en/latest/examples/unit-commitment.html) on building a problem instance.

-----

### Building the network in PyPSA
We start by importing PyPSA and creating a network with three time steps.
<details>
  <summary> TODO? </summary>
  hide code
</details>

```
import PyPSA
network = pypsa.Network(snapshots=range(3))
```
Then we have to add buses at which generators and loads are located and connect them with transmission lines. 
```
network.add("Bus","bus_1")
network.add("Bus","bus_2")
network.add(
    "Line",
    "line",
    bus0="bus_1"
    bus1="bus_2"
    s_nom=3,
)
```
We can now add the generators and loads to the buses.
```
network.add(
    "Generator",
    "gen_1"
    bus="bus_1",
    committable=True,
    p_min_pu=1,
    marginal_cost=15,
    p_nom=4,
)
network.add(
    "Generator",
    "gen_2",
    bus="bus_2",
    committable=True,
    p_min_pu=1,
    marginal_cost=10,
    p_nom=3,
)
network.add(
    "Load",
    "load_1",
    bus="bus_1",
    p_set=[2,3,2]
)
network.add(
    "Load",
    "load_2",
    bus="bus_3",
    p_set=[2,1,1]
)
```
The parameter `p_nom` sets the output of the generators and the `p_set` parameter sets a load at each time step for the respective bus. We don't have any minimal up or downtime and by setting `p_min_pu` we require generators to provide 100% of their power if they are committed.
The QUBO-based solvers ignore the `committable` flag and assume that every generator is committable. The `marginal_cost` keyword sets the cost of producing one unit of power.



### Converting a network to JSON

In order to submit this network to the service, we have to turn it into JSON-format. We can do that like this:
```
import json

network_xarray = network.export_to_netcdf()
network_dict = network_xarray.to_dict()
network_json = json.dump(network_dict)
```
Now you can pass that JSON to the service to solve it. A graphical representation of the problem looks like this.

INSERT_GRAPHIC

You can find the full code for generating the network below

<details>
    <summary> click to expand </summary>

```
import PyPSA
import json

# create empty network
network = pypsa.Network(snapshots=range(3))

# Add two busses
network.add("Bus","bus_1")
network.add("Bus","bus_2")
network.add(
    "Line",
    "line",
    bus0="bus_1"
    bus1="bus_2"
    s_nom=3,
)

# Add generators and loads
network.add(
    "Generator",
    "gen_1"
    bus="bus_1",
    committable=True,
    p_min_pu=1,
    marginal_cost=15,
    p_nom=4,
)
network.add(
    "Generator",
    "gen_2",
    bus="bus_2",
    committable=True,
    p_min_pu=1,
    marginal_cost=10,
    p_nom=3,
)
network.add(
    "Load",
    "load_1",
    bus="bus_1",
    p_set=[2,3,2]
)
network.add(
    "Load",
    "load_2",
    bus="bus_3",
    p_set=[2,1,1]
)

# transform to json
network_xarray = network.export_to_netcdf()
network_dict = network_xarray.to_dict()
network_json = json.dump(network_dict)
```

</details>

---

## Service Configuration

In order to choose the solver and configure it's parameters, the field `params` is used to pass another JSON-object. Because there are multiple solvers and each solvers has some unique parameters, there are a lot of possible options to specify in that field. Therefore we will go through the structure of the JSON object and explain how you can configure the various aspects of the different solvers.

The JSON-object contains these four fields: 
- `backend`, 
- `backend_config`, 
- `ising_interface` 
- `API_token`. 

They specify which solver is going to be used and it's configuration. The field `backend` contains a string denoting which solver to use and `backend_config` contains configuration parameters of that solver. Most of the solvers require the unit commitment problem to be cast as a [quadratic unconstrained binary optimization problem (QUBO)](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization). If a solver requires a QUBO formulation of the unit commitment problem, 
the field `ising_interface` specifies all relevant details on how the QUBO is built using another JSON-object.
At last, the field `API_token` contains a JSON-object that can be used to pass API tokens of services which are used by solvers that access quantum hardware.

### Supported Solvers
This service supports various solvers for obtaining solutions of the unit commitment problem.  Currently, the main solvers that can be used consist of:
- [Mixed Integer Linear Programming (GLPK)](https://en.wikipedia.org/wiki/Integer_programming)
- [Simulated Quantum Annealing](https://platform.planqk.de/algorithms/4ab6ed1f-9f5e-4caf-b0b2-59d1444340d1/)
- [Quantum Annealing](https://en.wikipedia.org/wiki/Quantum_annealing)
- [Quantum Approximation Optimization Algorithm](https://qiskit.org/textbook/ch-applications/qaoa.html)

The following solvers from [D-Waves ocean stack](https://docs.ocean.dwavesys.com/projects/system/en/stable/index.html) can also be chosen to solve the QUBO. Currently, you can't pass any parameters to them, which makes them unsuited for solving larger problems. 

- [Tabu search](https://docs.ocean.dwavesys.com/en/stable/docs_tabu/sdk_index.html)
- [Steepest descent](https://docs.ocean.dwavesys.com/en/stable/docs_greedy/sdk_index.html)
- [Hybrid solver](https://docs.ocean.dwavesys.com/en/stable/docs_hybrid/sdk_index.html)


The following table lists all supported solvers and how to choose them by passing the correct string to the `backend` field. 


| solver            | keyword      | description                                                                               | configuration keyword | uses QUBO | API token |
|-------------------|--------------|-------------------------------------------------------------------------------------------|-----------------------|-----------|-----------|
| sqa               | sqa          | performs simulated quantum annealing. Default solver if none is speficied                 | sqa_backend           | Yes       | None      |
| annealing         | classical    | performs simulated annealing                                                              | sqa_backend           | Yes       | None      |
| qaoa              | qaoa         | performs QAOA using IBM's qiskit by either simulating or accessing IBM's quantum computer | qaoa_backend          | Yes       | IBMQ      |
| glpk              | pypsa-glpk   | solves a mixed integer linear program obtained by pypsa using the GLPK solver             | pypsa_backend         | No        | None      |
| quantum annealing | dwave-qpu    | performs quantum annealing using d-waves quantum annealer                                 | dwave_backend         | Yes       | D-Wave    |
| tabu search       | dwave-tabu   | performs tabu search                                                                      | dwave_backend         | Yes       | None      |
| steepest decent   | dwave-greedy | performs steepest descent                                                                 | dwave_backend         | Yes       | None      |
| hybrid solver     | dwave-hybrid | uses d-waves hybrid solver in the cloud                                                   | dwave_backend         | Yes       | D-Wave    |

TODO: Remove config keyword because it is confusing?

Each solver also supports an additional configuration keyword which can also be used to passed parameters. 
If both the `backend` and the solver specific configuration field set the same parameter, the latter value takes precedence.
Since the main focus of this service lies on (simulated) quantum annealing and the MILP reference solution, configuring tabu search, steepest descent and the hybrid algorithm will be added later. 
Thus, choosing a solver adds the following entries the `params` dictionary:

```
{
    "backend": f"{keyword}",
    "backend_config": {}
    f"{configuration keyword}": {}
}
```

Because all solvers except the mixed integer linear program require a QUBO formulation, we will explain how to configure the QUBO model first.
A good model has few and distinct low energy states, which leads to a high probability of sampling a good solution when using a heuristic.

----

### Configuring the QUBO-model

The configuration of the model consists of two parts. First, you have to choose how network components, which can have various states, are represented as groups of variables. 
Then, you have to specify for each constraint which method is used to encode it into the QUBO using the aforementioned variables.

Each group of variables that represent a component has weights associated to them. These weights describe how much power
a variable represents. Then the sum of all weighted involved variables represent how much power is associated to that component. 
It depends on the type of the network component how it is interpreted in the context of different problem constraints. 
Choosing different methods for encoding network components changes how many variables are used and how these weights are determined.

#### QUBO representation of a transmission line

Variables encoding transmission line represent a fixed amount of directed power flow. Depending on the flow direction and the bus, this will act like a generator or load at each bus.
The direction of flow is indicated by the sign of the weights. In order to never exceed the limit of the transmission lines, the weights are set up in a way, that the absolute value of
any subtotal of weights is smaller than the capacity of the line.


The following table shows how to configure which transmission line representation to use when constructing the QUBO. The key in the `ising_interface` field is `line_representation` .
All values are rounded to integers.


| parameter value | description of line representation                                                   | additional information                                                                              |
|-----------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| fullsplit       | each variable has a weight of 1 or -1                                                | scales linearly with line capacity, which results in a high runtime                                 |
| customsplit     | compromise of minimizing number of variables and differences in magnitude of weights | only accepts capacity values between 1 and 4                                                        |
| cutpowersoftwo  | binary representation in each direction with a cut off at the highest order          | cheapest option with regards to number variables , but has huge differences in magnitude of weights |

#### QUBO representation of a generator

Variables encoding a generator represent fixed amounts of generated power. The sum of all weights is equal to the maximal power it can produce.

The unit commitment problem specifies a minimal power output for a generator. This can be archieved by transforming first-order interactions into second-order interactions with one qubit
singled out as a status qubit, which has a weight equal to the minimal power output.

The following table shows how generators can be encoded. The key is `generator_representation`.
All values are rounded to integers.

| paramter value        | description of generator representation        |
|-----------------------| ---------------------------------------------- |
| singleton             | uses a single qubit with weight equal to the maximal power
| integer_decomposition | binary decomposition of all integers in the range of maximal power. Ignores minimal power output


In total, choosing generator and transmission lines encodings add the following entries to the configuration:

```
{
    "ising_interface": {
        "line_representation": "cutpowersoftwo",
        "generator_representation": "singleton"
    }
}
```

### Encoding constraints

Because a QUBO is by definition unconstrained, we have to turn the unit commitment contraints into a QUBO formulation similar to the method of [lagrangian multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier).
For each constraint, the information how we transform it into a QUBO is written to the `ising_interface` field as another JSON-object. Each of these dictionaries have the key `scale_factor`
which sets the lagrange multiplier when calculating the complete QUBO.

The following table describes all supported constraints. Each other key that is not listed in the colum of keywords in the `ising_interface` field is ignored.

|  keyword             |  problem constraint description
| -------------------- | ----------------------------------------------------
| kirchhoff            | The kirchhoff constraint requires that the supplied energy is equal to the load at all buses
| marginal_cost        | The marginal costs is to be minimal among all feasible solutions


We will now go over the various constraints, and how we can choose different approaches to encode it into a QUBO. While all these approaches are technically a correct way, they have different limitations and properties.


#### Kirchhoff constraint

The kirchhoff constraint enforces that for a feasible solution, the load is equal to the supplied energy everywhere and at each time step. 
We model this as a sum of QUBOs, each one enforcing it at one bus for each bus. 
For each bus, the QUBO is constructed in a way that the objective function is the squared distance of the load at the bus to the supplied power.
Because this constraint is integral to the problem, it will always be added to the problem and only has the parameter `scale_factor`. 

For example, adding the kirchhoff constraint with a lagrange multiplier of `2.0` to the QUBO would require adding the following entry to the `ising_interface`:
```
{
    "ising_interface": {
        "kirchhoff": {
            "scale_factor": 2.0
        }
    }
}
```


#### Marginal Costs

The standard method of encoding the objective function of an optimization problem into a QUBO is to formulate a QUBO which, evaluated for any state, is equal to the objective function.
This direct encoding runs into a problem once we add additional constraints. Because a QUBO is by definition unconstrained, the lagrange multipliers of the constraints have to be large enough
that it is never beneficial to violate a constraint in order to minimize the objective function. This makes the QUBO hard to solve because fulfilling the constraint overshadows the optimization
of the objective function.

Therefore, this service implements the direct encoding, but also a second approach to describe the minimization of the marginal costs as a QUBO. The basic idea is that instead of encoding the objective
function, we encode the squared distance of the objective function to an estimation. 
The caveat is that the QUBO is technically only equivalent to the unit commitment problem if the estimation is close enough to the actual mininum.
However, a solution of such a QUBO still gives us information how much the kirchhoff constraint had to be violated to get close enough to the estimation which lets us improve it.

The approach based on the estimation of the costs also has other benefits. First, the quadratic growth of the squared distance ensures that the constraint will always outscale the impact of the lagrange multipliers.
The other benefit is that good estimations lead to well conditioned problems because the kirchhoff constraint and the marginal costs can be optimally solved by the same solution.
This allows us to choose lagrange multipliers with similar orders of magnitude without the optimization of the marginal costs superceding the optimization of the kirchhoff constraint. 

The following table describes various strategies to encode the marginal cost into a QUBO using the key `strategy`.

|    configuration value              |                 strategy description                                 
| ----------------------------------- | ------------------------------------------------------------------
| marginal_as_penalty                 | direct translation from cost to energy as first order interactions                                  
| global_cost_square                  | squared distance of total cost to an estimation                    
| global_cost_square_with_slack       | squared distance of total cost to an estimation with slack variables

TODO: ensure lower bound??
TODO: option to add estimation out of the offset

##### Configuring the estimation

TODO: check how the reference value is calculated
 
The parameter for configuring the estimation is `offset_estimation_factor`. When constructing the QUBO for the marginal costs, the marginal costs of each generator
are offset by the product of that factor and the cost of the most efficient generator. 
Such an offset doesn't change the solution of the unit commitment problem because all feasible solution are offset by the same value.

The estimation used to encode the marginal cost assumes that the cost of the optimal solutin is zero with respect to the offset marginal costs.
Thus the estimated value for some `offset_estimation_factor` is the product of it, the marginal cost of the most efficient generator and the total load of the network.


The strategy `global_cost_square_with_slack` has three more options on top of the `offset_estimation_factor`. It adds slack variables that act like generators that reduce marginal costs, but produce
no power. This changes the estimation from a constant value to that constant plus the number represented by the slack variables.
k

The following table describes the parameters that are used to describe these slack variables.

|  keyword                  |  description 
| ------------------------  | ------------------------------------------------------  
|  slack_type               | a string that specifies by which rule the slack weights are generated. 
|  slack_size               | an integer that determines the number of slack variables
|  slack_scale              | a float the scales all slack variable weights

The only supported value for `slack_type` is `binary_power`, which configures weights of slack variables as ascending powers of two.  In conjuction with `slack_size`, which specifies how many slack variables are created, the slack in marginal costs can be set up as as any fixed length binary number. The `slack_scale` works similar to `scale_factor` scaling the weights of the slack variables.

In total, an exmaple of a JSON-object for configuring the QUBO looks like this. 

```
{
    "ising_interface": {
        "generator_representation": "singleton",
        "line_representation": "cutpowersoftwo",

        "kirchhoff": {
            "scale_factor": 1.0
        },
        "marginal_cost": {
            "strategy": "global_cost_square_with_slack",
            "scale_factor": 0.3,
            "offset_estimation_factor": 1.0
            "slack_type": "binary_power",
            "slack_scale": 0.1,
            "slack_size": 3,
        }
    }
}
```

Since almost every solver this service provides solves the QUBO that is specified by the `ising_interface`, we now want to briefly explain how you can choose good parameters. It is important to choose a well-behaved QUBO because this keeps the required runtime, that a solver needs to solve it well, low. Finding a feasible solution that just satisfies the kirchhoff constraint is very simple. However, when we try to solve two problems simultaneously, in this case the kirchhoff constraint and the marginal cost, finding a good solutions becomes a lot harder. The issue is that, compared to just the kirchhoff constraint, the set of good solutions is significantly smaller. Another issue is that improvements in one subproblem tend to incur costs in the other. Because the kirchhoff constraint is mandatory, you also have to make sure that the solution solves it optimally, which tends to lead to huge differences in scale. Difference in scales of subproblems also lead to bad results because the bigger problem overshadows the smaller, which in turn gets solved badly due to the gradient with respect to it being so small.

You can think of the solver as a stochastic process to explain this. For a given QUBO, solving it with a specific quality can be thought of as random sampling from the distribution of all states.  This sample has the property that the distance to the optimal solution with regards to the QUBO's cost is limited. A QUBO is well-behaved if the set of states that are close enough to that optimal solution is very small, which would result in good convergence. However, huge difference in scales inflate this set. If the magnitude of the scales is similar, this means that both subproblems are solved similarly well. However, the kirchhoff constraint has to be solved optimally.

In order to get around this, the idea is solve a series of QUBOs, with each previous solution allowing you to tweak the QUBO in a way that brings the two subproblems closer together. This is what the estimation based strategies are for, with the estimation serving as a means to get around the issue of differences in scale messing up the quality of the solution.

### Backend Config

At last we describe the different solvers that this service provides, how to use them and which options you have for configuring them. Each solver that you can choose solves a QUBO, so the difference between them is how they solve it. The only exception are the solvers that solve a mixed integer linear program like glpk. We now go through the different solvers and describe how you can use `backend_config` to configure it.

The default QUBO-based solver is simulated quantum annealing.

#### Simulated Quantum Annealing
This solver solves a QUBO using a monte carlo simulation of quantum annealing to find an optimal solution.  You can find more information on simulated quantum annealing [here](https://platform.planqk.de/algorithms/4ab6ed1f-9f5e-4caf-b0b2-59d1444340d1/) and on quantum annealing [here](https://platform.planqk.de/algorithms/786e1ff5-991e-428d-a538-b8b99bc3d175/).  More on the used solver itself can be found [here](https://platform.lanqk.de/services/ff5c0cdd-4a87-4473-8086-cb658a9f85a2) In this particular case, this service uses the same implementation as the SQA service, but also provides the tools to construct the QUBO, as well as interpret a solution as a state of the network.

The following table shows which  parameters can be adjusted.

parameter                 |  Default Value    |    Description                                                                                             |  Runtime Impact            
------------------------- | ----------------- |  --------------------------------------------------------------------------------------------------------  |  -----------------  
transverse_field_schedule | "[8.0,0.0]"       |  The transverse field strength schedule. It describes the interaction strength between trotter slices      |  None
temperature_schedule      | "[0.1,iF,0.0001]" |  The temperature schedule. At higher temperatures, qubit states are more likely to change                  |  None
trotter                   | 32                |  The number of trotter slices. They are an approximation of the quantum state of quantum annealing         |  grows linearly
optimization_cycles       | 20                |  The number of discrete time steps used to simulate the continuous evolution of the quantum state          |  grows linearly
seed                      | N/A               |  The seed used by the random number generator for the solver. Unless explicitly set, the seed is random    |  None
time_limit                | None              |  An upper limit of the estimated runtime. The service will fail right away if this limit is exceeded       |  None or stops the service

You can see that the first two parameters both describe a schedule. Thus, we have to explain what a schedule is and how it affects the solver. Simulated quantum annealing simulates the continous evolution of a quantum state in multiple discrete time steps. The two schedules determine the strength of two aspects of the simulation at each time step. Similar to classical annealing, the temperature is a measure of how much change can happen in one time step, while the transverse field is a measure of interaction strength between different trotter slices.

#### Specifying a Schedule

A schedule is a function $f \colon [0,1] \rightarrow \mathbb{R}$ of the unit interval.  If the simulation consists of $n+1$ steps with the zeroth step $t_0 = 0$ and last step $t_n = 1$, the strength of a field according to that schedule at the $i$-th step is given as $f(\frac{i}{n})$. The string that describes the schedule is simply a shorthand for refering to a member of a  family of functions that can be used as schedules. You can also concatenate different schedules in which case they get evenly  distributed over the unit interval. The following table gives an overview over admissable strings.

Schedule String    |  Function                                                                                                        | Description
----------------   | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------
[10,1] or [10,l,1] |  $p \mapsto 10 - 9 \cdot p$                                                                                      | a linear ramp from 10 to 1 
[a,iF,b]           |  $p \mapsto \frac{a \cdot b}{b + (a - b) \cdot p}$                                                               | This is fast at the beginning and slow towards the end
[a,iS,b]           |  $p \mapsto a + b - \frac{a \cdot b}{a - (a - b) \cdot p}$                                                       | This is slow at the beginning and fast towards the end
[a,sS,b]           |  $p \mapsto a + (b - a) \cdot p^2$                                                                               | N/A
[a,sF,b]           |  $p \mapsto b + (a - b) \cdot (p-1)^2$                                                                           | N/A
[10,l,2,2,l,1]     |  $p \mapsto \begin{cases} 10 - 16 \cdot p & p \leq \frac{1}{2} \\ 3 - 2 \cdot p & p \geq \frac{1}{2} \end{cases}$| first a linear ramp from 10 to 2 and then in the same time a linear ramp from 2 to 1

Any string that adheres to the rules above is a valid value for the temperature or transverse field schedule. As general advice, both the number of trotter slices, as well as time steps increase the solution quality, but also increase the runtime linearly. Increasing them however has dimininishing returns on solution quality, with the number of time steps dminishing faster.

```
{
    "sqa_backend": {
        "transverse_field_schedule": "[8.0,0.0]",
        "temperature_schedule": "[0.1,iF,0.0001]",
        "trotter_slices": 512,
        "optimization_cycles": 128
    }
}
```

#### Quantum Annealing
Quantum Annealing uses D-Wave's quantum hardware to perform quantum annealing. For that, it uses the advantage 4.1 quantum annealer.  This requires an API token. Due to to hardware limitations, problem size is significantly limited and solutions aren't as accurate as other solvers. You can find more information on the quantum annealer [here](https://docs.ocean.dwavesys.com/en/stable/overview/qpu.html) and more information on the parameters that the annealer accepts [here](https://docs.dwavesys.com/docs/latest/c_solver_parameters.html). The service builds the QUBO and commits it to d-wave's cloud service.  For this solver, a d-wave API token is mandatory.

The most important parameters for this solvers are listed in the following table:

|  parameter       |  description  
| ---------------- | ------------------------------------
| annealing_time   | duration of a single annealing run in microseconds  
| num_reads        | number of repetitions of quantum annealing
| chain_strength   | the strength of chains used to embed the QUBO onto the working graph of the annealer
| timeout          | time in seconds that the embedding algorith has to embed the QUBO onto the working graph before it fails

Due to noise in the samples, this solver also performs simple flow optimization to improve the results. Both annealing time and number of reads improve the solution. In general more annealing runs are better than annealing longer. The chain strength has to adjusted to the problem at hand. If the strength is two small, breaks in chains break the theoretical model of the problem which leads to bad solutions.  If the chain strength is to high, it's magnitude overshadows the optimization of the actual problem, once again, leading to bad results.  In total, an example for the configuration of the quantum annealer looks like this:

```
"dwave_backend": {
    timeout: 100
    annealing_time: 50
    num_reads: 200
    chain_strength: 70
}
```

Instead of `dwave_backend`, you can also use `backend_config` as the key for the four configuration parameters.


#### Quantum Approximation Optimization Algorithm (QAOA)
Right now, the QAOA solver is broken and using it will not return a result.

The quantum approximation optimization algorithm build a parametrized quantum circuit to solve the problem. Then, it uses a classical optimizer to adjust the parameters so the circuit better fits the problem. You can find more information on QAOA and how it works [here](https://qiskit.org/textbook/ch-applications/qaoa.html). For the implementation of this service, you can choose to either simulate the circuit or access IBM's quantum computer to run it. If you want to use a quantum computer or the noise model of a quantum computer, you have to provide an API token.

The problem Hamiltonian which is used to build the circuit can easily be obtained by transforming the QUBO that encodes the unit commitment problem into an equivalent ising spin-glass problem. Due to circuit-based architectures being extremely limited in size and a simulation of quantum circuits being, the problem size for this solver is also very limited.

The following Table describes the various parameters that QAOA accepts.

|  parameter name       |  description                                                                                         |  possible values
| --------------------  |  --------------------------------------------------------------------------------------------------  |  ----------------
|  shots                |  number of circuit evaluations when sampling the circuit during optimization                         |  a natural number up to 20000
|  simulate             |  boolean for deciding wether to use a simulator or submitting problems to quantum hardware           |  True or False
|  simulator            |  a string specify which simulator use for the quantum circuit if simulate is set to True             |  qasm_simulator, aer_simulator, statevector_simulator, aer_simulator_statevector
|  noise                |  boolean for deciding wether to use a noise model or not                                             |  True of False
|  max_iter             |  number of steps the classical optimizer uses to optimize the parametrization                        |  an integer
|  repetitions          |  number of independent optimiziation runs with different initial angle values                        |  an integer
|  classical_optimizer  |  a string specifying the classical optimizer used for optimizing parametrization                     |  SPSA, COBYLA, ADAM
|  supervisior_type     |  a string describing which method to use for angle initialization                                    |  RandomOrFixed, GridSearch
|  initial_guess        |  a list of values used in choosing inital angles of the various layers of the quantum circuit        |  depends on the supervisior_type
|  range                |  the range of values to be used when randomly initializing an angle                                  |  a float
  
Reading the description of QAOA, the impact of the first few parameters is clear. Regarding the simulator choice, you can find more information at qiskit's [documentation](https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html). The last three parameters are details of our implementation and require more information.

Over the course of the algorithm, QAOA iteratively improves the parametrization of the quantum circuit to find a circuit which fits the problem well. One issue with this approach that the initial parametrization has a huge impact because the optimization problem has a lot of local minima. In order to adress it, the parameter `repetitions` allows us to run QAOA multiple times with different initial values. However, you still have to specify which initial values to use which is what the last three parameters is for.

Because the parametrization of the quantum circuit can be described by a tuple of floating values, we provide two different methods for generating these tupels in the `supervisior_type` field. Each of these require a different type of argument for the value of `inital_guess`

The option `RandomOrFixed` requires a list with values either being floating values or the string `"rand"`. When QAOA requires a tuple of initial angles, this list is returned after substituting each entry `"rand"` with a random floating value. The intervall in which the randomly chosen values lie is given by the value of `range`. For a parameter value `x`, the intervall from which a random number is drawn is given as `(-x, x)` with the default value being `3.14` if none is passed in the configuration. Then, for every possible combination, a QAOA run is started.

The option `GridSearch` specifies a regular set of floating values which are possible as initial values for each layer of the quantum circuit. A set of floating values for a layer is specified as a JSON-object with the keys `lower_bound`, `upper_bound` and `num_gridpoints`. The values for each field are floats, with initial values being in the intervall given by the bounds.

The following JSON-object is an example for the a configuaration of QAOA using random initialization of some angles:
```
{
    "qaoa_backend": {
        shots: 200,
        simulate: "True",
        noise: "False",
        simulator: "aer_simulator",
        max_iter: 20,
        repetitions: 10,
        classical_optimizer: "COBYLA",
        supervisior_type: "RandomOrFixed",
        initial_guess: ["rand", "rand"],
        range: 3
    }
}
```
Using the option `GridSearch`, the configuration would look like this

```
{
    "qaoa_backend": {
        shots: 200,
        simulate: "True",
        noise: "False",
        simulator: "aer_simulator",
        max_iter: 20,
        classical_optimizer: "COBYLA",
        supervisior_type: "GridSearch",
        initial_guess: [
            {
                lower_bound: -2
                upper_bound: 2
                num_gridpoints: 3
            },
            {
                lower_bound: -2
                upper_bound: 2
                num_gridpoints: 3
            }
        ]
    }
}
```

#### Mixed Integer Linear Programming
The [mixed integer linear programming approach](https://en.wikipedia.org/wiki/Linear_programming) uses pypsa's native method to cast the problem into a mixed integer linear program and solve it using GLPK. Since is just a wrapper for pypsa native solving method, this has just one option for the maximal duration of the optimization.

|   configuration key  |  impact            
| -------------------- | ------------------ 
| timeout              | time in seconds after which the optimization will be stopped and the current best result will be returned

An example for configuring this solver is very short. Due to the nature of the problem, it is easy to find a decent feasible solution fast, but finding a close to optimal solutions takes a lot of times for bigger problems.

```
{
    "pypsa_backend": {
        "timeout": 10
    }
}
```

#### API-token
Using D-Wave's quantum hardware or using an IBM quantum computer for qaoa requires an API token. For d-wave services, you have to use the key `dwave_API_token`, and for ibm, the key is  `IMBQ_API_token`.

```
"API_token": {
    "IBMQ_API_token": "Your IMBQ API token",
    "dwave_API_token": "Your d-wave API token"
}
```

### Output format

After a succesful calculation, this service returns a JSON object. This JSON-object has a field for storing all relevant result details. Is also saves the various parameters used to generate the result in order to give some context to the result The returned JSON has the following fields:

- `results`: this contains information about the solution
- `config`: the various configuration parameters that were passed to the solver
- `network`: the serialized network which's unit commitment problem was solved
- `start_time`, `end_time`: the start and end time in `yyyy-mm-dd-hh-mm-ss` format

For each solver, the following values are in the `results` object for each solver The following table explains all result data that all solvers return.

|  keyword                   |     description  
| -------------------------- | ---------------
| total_cost                 |  Cost of the QUBO representing the unit commitment problem including constants.
| kirchhoff_cost             |  sum of all squared deviations from satisfiying the kirchhoff constraint
| power_imbalance            |  sum of all linear absolute deviations from satisfiying the kirchhoff constraint
| total_power                |  total power generated by the solution
| marginal_cost              |  total marginal costs incurred by the solution
| individual_kirchhoff_cost  |  JSON object with stringified (bus, snapshot) tupels as keys, and the deviation from the kirchhoff constraint at that bus and snapshot as values
| unit_commitment            |  JSON object with stringified (generator, snapshot) tupels as keys, and the percentange of the maximal power output of that generator in the solution at that snapshot as values
| powerflow                  |  JSON object with stringified (line, snapshot) tupels as keys, and the powerflow including the direction as the sign using that line at that snapshot as values

The solvers also write more data with information about their optimization run. Because this data is not relevant for the solution itself and highly dependant on the solver, we won't explain that output.

At last, we provide a small example that can be used to test the solvers. In the example below we have used a network consisting of two buses with one generator at each bus. At the first bus, the generator produces 3 MW and the load is 1 MW. At the other bus, the generator produces 2 MW and the load is 2 MW. The two buses are connected by a transmission line with a total capacity of 2 MW.  Thus, the optimal solution is two commit the first generator and transport 2 MW to the second bus. As you can see in the example below, even such a simple network results in a relatively huge serialized object so it strongly recommended to build the network in PyPSA and then serialized.  For better readability, the network data is the second entry with `params` being the first. There you also have an overview of all the different solvers. Even though the field `backend_config` is empty, it is functionally the same to write it into the corresponding `_backend` JSON. Which solver is chosen in only determined by the field `backend`

<details>
    <summary> JSON containing valid input for network and parameters </summary>

{
  "params": {
    "API_token": {
      "IBMQ_API_token": "",
      "dwave_API_token": ""
    },
    "backend": "sqa",
    "ising_interface": {
      "formulation": "fullsplit"
    },
    "qaoa_backend": {
      "shots": 500,
      "simulate": true,
      "noise": false,
      "simulator": "aer_simulator",
      "initial_guess": [
        "rand",
        "rand",
        "rand",
        "rand"
      ],
      "max_iter": 20,
      "repetitions": 16,
      "classical_optimizer": "COBYLA"
    },
    "sqa_backend": {
      "transverse_field_schedule": "[8.0,0.0]",
      "temperature_schedule": "[0.1,iF,0.0001]",
      "trotter_slices": 128,
      "optimization_cycles": 100
    },
    "dwave_backend": {
      "annealing_time": 30,
      "num_reads": 200,
      "chain_strength": 60,
      "timeout": 60,
    },
    "pypsa_backend": {
      "timeout": 10
    },
    "backend_config": {
    }
  },
  "data": {
    "coords": {
      "snapshots": {
        "dims": [
          "snapshots"
        ],
        "attrs": {},
        "data": [
          0
        ]
      },
      "investment_periods": {
        "dims": [
          "investment_periods"
        ],
        "attrs": {},
        "data": []
      },
      "generators_i": {
        "dims": [
          "generators_i"
        ],
        "attrs": {},
        "data": [
          "Gen1",
          "Gen2"
        ]
      },
      "buses_i": {
        "dims": [
          "buses_i"
        ],
        "attrs": {},
        "data": [
          "bus1",
          "bus2"
        ]
      },
      "loads_i": {
        "dims": [
          "loads_i"
        ],
        "attrs": {},
        "data": [
          "load1",
          "load2"
        ]
      },
      "lines_i": {
        "dims": [
          "lines_i"
        ],
        "attrs": {},
        "data": [
          "line1"
        ]
      }
    },
    "attrs": {
      "network_name": "",
      "network_pypsa_version": "0.19.3",
      "network_srid": 4326
    },
    "dims": {
      "snapshots": 1,
      "investment_periods": 0,
      "generators_i": 2,
      "buses_i": 2,
      "loads_i": 2,
      "lines_i": 1
    },
    "data_vars": {
      "snapshots_snapshot": {
        "dims": [
          "snapshots"
        ],
        "attrs": {},
        "data": [
          "now"
        ]
      },
      "snapshots_objective": {
        "dims": [
          "snapshots"
        ],
        "attrs": {},
        "data": [
          1
        ]
      },
      "snapshots_generators": {
        "dims": [
          "snapshots"
        ],
        "attrs": {},
        "data": [
          1
        ]
      },
      "snapshots_stores": {
        "dims": [
          "snapshots"
        ],
        "attrs": {},
        "data": [
          1
        ]
      },
      "investment_periods_objective": {
        "dims": [
          "investment_periods"
        ],
        "attrs": {},
        "data": []
      },
      "investment_periods_years": {
        "dims": [
          "investment_periods"
        ],
        "attrs": {},
        "data": []
      },
      "generators_bus": {
        "dims": [
          "generators_i"
        ],
        "attrs": {},
        "data": [
          "bus1",
          "bus2"
        ]
      },
      "generators_p_nom": {
        "dims": [
          "generators_i"
        ],
        "attrs": {},
        "data": [
          1.0,
          3.0
        ]
      },
      "generators_marginal_cost": {
        "dims": [
          "generators_i"
        ],
        "attrs": {},
        "data": [
          5.0,
          5.0
        ]
      },
      "loads_bus": {
        "dims": [
          "loads_i"
        ],
        "attrs": {},
        "data": [
          "bus1",
          "bus2"
        ]
      },
      "loads_p_set": {
        "dims": [
          "loads_i"
        ],
        "attrs": {},
        "data": [
          2.0,
          1.0
        ]
      },
      "lines_bus0": {
        "dims": [
          "lines_i"
        ],
        "attrs": {},
        "data": [
          "bus1"
        ]
      },
      "lines_bus1": {
        "dims": [
          "lines_i"
        ],
        "attrs": {},
        "data": [
          "bus2"
        ]
      },
      "lines_x": {
        "dims": [
          "lines_i"
        ],
        "attrs": {},
        "data": [
          0.0001
        ]
      },
      "lines_s_nom": {
        "dims": [
          "lines_i"
        ],
        "attrs": {},
        "data": [
          2.0
        ]
      }
    }
  }
}

</details>
