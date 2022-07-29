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

- [Output format](#Output-format)

# Input format

The input for this service is a JSON-object. This object has the two fields `data` and `params` to specifiy the problem and which solver to use respectively. 
The field `data` contains a serialized [PyPSA network](https://pypsa.readthedocs.io/en/latest/).
The other field `params` contains the name of the solver to be used and configuration parameters for that solver.

First we go through the steps of building a PyPSA network that we can then pass to the service.
The PyPSA documentation also contains an [example](https://pypsa.readthedocs.io/en/latest/examples/unit-commitment.html) on building a problem instance.

-----

### Building the network in PyPSA
We start by importing PyPSA and creating a network with three time steps.

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
    "gen_1",
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
    bus="bus_2",
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

Now you can pass that JSON to the service to solve it. 
You can find the full code for generating the network below:

<details>
    <summary> click to expand </summary>

```
import pypsa
import json

# create empty network
network = pypsa.Network(snapshots=range(3))

# Add two busses
network.add("Bus","bus_1")
network.add("Bus","bus_2")
network.add(
    "Line",
    "line",
    bus0="bus_1",
    bus1="bus_2",
    s_nom=3,
)

# Add generators and loads
network.add(
    "Generator",
    "gen_1",
    bus="bus_1",
    committable=True,
    p_min_pu=1,
    marginal_cost=15,
    p_nom=4
)
network.add(
    "Generator",
    "gen_2",
    bus="bus_2",
    committable=True,
    p_min_pu=1,
    marginal_cost=10,
    p_nom=3
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
    bus="bus_2",
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

| solver            | keyword      | description                                                                               | uses QUBO | API token |
|-------------------|--------------|-------------------------------------------------------------------------------------------|-----------|-----------|
| sqa               | sqa          | performs simulated quantum annealing. Default solver if none is speficied                 | Yes       | None      |
| annealing         | classical    | performs simulated annealing                                                              | Yes       | None      |
| qaoa              | qaoa         | performs QAOA using IBM's qiskit by either simulating or accessing IBM's quantum computer | Yes       | IBMQ      |
| glpk              | pypsa-glpk   | solves a mixed integer linear program obtained by pypsa using the GLPK solver             | No        | None      |
| quantum annealing | dwave-qpu    | performs quantum annealing using d-waves quantum annealer                                 | Yes       | D-Wave    |
| tabu search       | dwave-tabu   | performs tabu search                                                                      | Yes       | None      |
| steepest decent   | dwave-greedy | performs steepest descent                                                                 | Yes       | None      |
| hybrid solver     | dwave-hybrid | uses d-waves hybrid solver in the cloud                                                   | Yes       | D-Wave    |

Since the main focus of this service lies on (simulated) quantum annealing and the MILP reference solution, configuring tabu search, steepest descent and the hybrid algorithm will be added later. 
By adding the example below to the `params` field, you can choose the simulated quantum annealing solver.

```
{
    "backend": "sqa",
    "backend_config": {}
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

#### Limiting snapshots

In order to limit the size of the QUBO, this service has an option to limit the number of snapshots that are considered for the unit commitment problem
This can be done by passing an integer to the field `snapshots`.
For example, adding the following entry to the configuration would limit the unit commitment problem to the first two snapshots.

```
{
    "ising_interface": {
        "snapshots": 2
    }
}
```

----

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
| single_qubit             | uses a single qubit with it's weight equal to the maximal power
| integer_decomposition | binary decomposition of all integers in the range of maximal power. Ignores minimal power output

In total, choosing generator and transmission lines encodings add the following entries to the configuration:

```
{
    "ising_interface": {
        "line_representation": "cutpowersoftwo",
        "generator_representation": "single_qubit"
    }
}
```

----

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

##### Configuring the estimation

The parameter for configuring the estimation is `offset_factor`. When constructing the QUBO for the marginal costs, the marginal costs of each generator
are offset by the product of that factor and the cost of the most efficient generator. 
Such an offset doesn't change the solution of the unit commitment problem because all feasible solution are offset by the same value.

The estimation used to encode the marginal cost then assumes that the cost of the optimal solutin is zero with respect to the offset marginal costs.
Thus the estimated value for some `offset_factor` is the product of it, the marginal cost of the most efficient generator and the total load of the network.

The strategy `global_cost_square_with_slack` has three more options on top of the `offset_factor`. It adds slack variables that act like generators that reduce marginal costs, but produce
no power. This changes the estimation from a constant value to that constant plus the number represented by the slack variables.

The following table describes the parameters that are used to describe these slack variables.

|  keyword                  |  description 
| ------------------------  | ------------------------------------------------------  
|  slack_type               | a string that specifies by which rule the slack weights are generated. 
|  slack_size               | an integer that determines the number of slack variables
|  slack_scale              | a float the scales all slack variable weights

The only supported value for `slack_type` is `binary_power`, which configures weights of slack variables as ascending powers of two.  In conjuction with `slack_size`, which specifies how many slack variables are created, the slack in marginal costs can be set up as as any fixed length binary number. The `slack_scale` works similar to `scale_factor` scaling the weights of the slack variables.

In total, an example of a JSON-object for configuring the QUBO looks like this. 

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
            "offset_factor": 1.0,
            "slack_type": "binary_power",
            "slack_scale": 0.1,
            "slack_size": 3,
        }
    }
}
```

----

### Solver Configuration

At last we describe the different solvers that this service provides, how to use them and which options you have for configuring them using the field `backend_config`.
If no solver has been specified, simulated quantum annealing will be used.
For each solver, a runtime duration is estimated and if exceeded, the optimization will not be performed.

----

#### Simulated Quantum Annealing
This solver solves a QUBO using a monte carlo simulation of quantum annealing to find an optimal solution. You can find more information on simulated quantum annealing [here](https://platform.planqk.de/algorithms/4ab6ed1f-9f5e-4caf-b0b2-59d1444340d1/) and on quantum annealing [here](https://platform.planqk.de/algorithms/786e1ff5-991e-428d-a538-b8b99bc3d175/). More on the solver itself can be found [here](https://platform.lanqk.de/services/ff5c0cdd-4a87-4473-8086-cb658a9f85a2). 
This service uses the same implementation as the SQA service, but also provides the translation of the unit commitment problem to a QUBO problem.

The following table shows which  parameters can be adjusted.

parameter                 |  Default Value    |    Description                                                                                             |  Runtime Impact            
------------------------- | ----------------- |  --------------------------------------------------------------------------------------------------------  |  -----------------  
trotter                   | 32                |  The number of trotter slices. They are an approximation of the quantum state of quantum annealing         |  grows linearly
optimization_cycles       | 20                |  The number of discrete time steps used to simulate the continuous evolution of the quantum state          |  grows linearly
transverse_field_schedule | "[8.0,0.0]"       |  The transverse field strength schedule. It describes the interaction strength between trotter slices      |  None
temperature_schedule      | "[0.1,iF,0.0001]" |  The temperature schedule. At higher temperatures, qubit states are more likely to change                  |  None
seed                      | N/A               |  The seed used by the random number generator for the solver. Unless explicitly set, the seed is random    |  None

The first two parameters control how accurate the simulation is. This determines the quality of the solution, but also the runtime.
The next two parameters describe a schedule. Thus, we explain what a schedule is and how it affects the solver. 
Simulated quantum annealing simulates the continous evolution of a quantum state in multiple discrete time steps. A schedule determines the strength of a field in the simulation at each time step.
In our case, this is the temperature and the transverse field. The former is a measure of how much qubits can change their state, while the latter is the interaction strength between different trotter slices.

##### Specifying a Schedule

A schedule is a function $f \colon [0,1] \rightarrow \mathbb{R}$ of the unit interval.  If the simulation consists of $n+1$ steps with the zeroth step $t_0 = 0$ and last step $t_n = 1$, the strength of a field according to that schedule at the $i$-th step is given as $f(\frac{i}{n})$.
You can configure a schedule by passing a string, which are shorthands for refering to a member of a  family of functions.
You can also concatenate different schedules in which case they get evenly distributed over the unit interval. The following table gives an overview over all admissable strings.

Schedule String    |  Function                                                                                                        | Description
----------------   | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------
"[10,1]" or "[10,l,1]" |  $p \mapsto 10 - 9 \cdot p$                                                                                      | a linear ramp from 10 to 1 
"[a,iF,b]"           |  $p \mapsto \frac{a \cdot b}{b + (a - b) \cdot p}$                                                               | This is fast at the beginning and slow towards the end
"[a,iS,b]"           |  $p \mapsto a + b - \frac{a \cdot b}{a - (a - b) \cdot p}$                                                       | This is slow at the beginning and fast towards the end
"[a,sS,b]"           |  $p \mapsto a + (b - a) \cdot p^2$                                                                               | A quadratic ramp from a to b that starts slow 
"[a,sF,b]"           |  $p \mapsto b + (a - b) \cdot (p-1)^2$                                                                           | A quadratic ramp from a to b that starts fast
"[10,l,2,2,l,1]"     |  $p \mapsto \begin{cases} 10 - 16 \cdot p & p \leq \frac{1}{2} \\ 3 - 2 \cdot p & p \geq \frac{1}{2} \end{cases}$| first a linear ramp from 10 to 2 and then in the same time a linear ramp from 2 to 1

Any string that adheres to the rules above is a valid value for the temperature or transverse field schedule. As general advice, both the number of trotter slices, as well as time steps increase the solution quality, but also increase the runtime linearly. Increasing them however has dimininishing returns on solution quality, with the number of time steps dminishing faster.

In total, a valid configuration for simulated quantum annealing looks like this.
```
{
    "backend_config": {
        "transverse_field_schedule": "[8.0,0.0]",
        "temperature_schedule": "[0.1,iF,0.0001]",
        "trotter_slices": 512,
        "optimization_cycles": 128
    }
}
```

----

#### Quantum Annealing

Quantum Annealing uses D-Wave's advantage 4.1 quantum annealer to perform quantum annealing. Due to the hardware's limitations, problem size is significantly limited and solutions aren't as accurate as other solvers. You can find more information on the quantum annealer [here](https://docs.ocean.dwavesys.com/en/stable/overview/qpu.html) and more information on the parameters that the annealer accepts [here](https://docs.dwavesys.com/docs/latest/c_solver_parameters.html). The service builds the QUBO and commits it to D-Wave's cloud service. This solver requires a D-Wave API token.

Only the parameters in the following table will be passed to the solver. The optimization itself doesn't have a time limit, but if the embedding onto the annealer's working graph takes to long, this service will cancel it.

|  parameter       |  description  
| ---------------- | ------------------------------------
| annealing_time   | duration of a single annealing run in microseconds  
| num_reads        | number of repetitions of quantum annealing
| chain_strength   | the strength of chains used to embed the QUBO onto the working graph of the annealer

The chain strength has to adjusted to the problem at hand. If the strength is too small, breaks in chains invalidate the theoretical model of the problem which leads to bad solutions.
If the chain strength is too high, it's magnitude overshadows the optimization of the actual problem, once again leading to bad results. 

Because the quantum annealer is noisy, this solver also performs simple flow optimization to improve the results. Both the annealing time and number of reads improve the solution but the impact of the former hits a plateau very quickly. 

In total, an example for the configuration of the quantum annealer looks like this:

```
"backend_config": {
    timeout: 100,
    annealing_time: 50,
    num_reads: 200,
    chain_strength: 70
}
```

----

#### Quantum Approximation Optimization Algorithm (QAOA)

The quantum approximation optimization algorithm build a parametrized quantum circuit to solve the problem. Then, it uses a classical optimizer to adjust the parameters so the circuit better fits the problem. You can find more information on QAOA and how it works [here](https://qiskit.org/textbook/ch-applications/qaoa.html). For the implementation of this service, you can choose to either simulate the circuit or access IBM's quantum computer to run it. If you want to use a quantum computer or the noise model of a quantum computer, you have to provide an API token.

The problem Hamiltonian which is used to build the circuit can easily be obtained by transforming the QUBO that encodes the unit commitment problem into an equivalent ising spin-glass problem. Due to circuit-based architectures being extremely limited in size and a simulation of quantum circuits being, the problem size for this solver is also very limited.

The following Table describes the various parameters that QAOA accepts.

|  parameter name       |  description                                                                                         |  possible values
| --------------------  |  --------------------------------------------------------------------------------------------------  |  ----------------
|  shots                |  number of circuit evaluations when sampling the circuit during optimization                         |  a natural number up to 20000
|  max_iter             |  number of steps the classical optimizer uses to optimize the parametrization                        |  an integer
|  classical_optimizer  |  a string specifying the classical optimizer used for optimizing parametrization                     |  SPSA, COBYLA, ADAM
|  simulate             |  boolean for deciding wether to use a simulator or submitting problems to quantum hardware           |  True or False
|  simulator            |  a string to specify which simulator to use for the quantum circuit if simulate is set to True       |  qasm_simulator, aer_simulator, statevector_simulator, aer_simulator_statevector
|  noise                |  boolean for deciding wether to use a noise model or not                                             |  True of False


Reading the description of QAOA and the description column in the should suffice for configuring these parameters. Regarding the simulator choice, you can find more information at qiskit's [documentation](https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html).

What is left is the choice of the initial parameters. Because the initial choice has a huge impact on the subsequent optimization, this service implements two strategies for choosing them over multiple independent runs. These two can be chosen by either passing the value `random_or_fixed` or `grid_search` to the field `strategy`. 

Because the quantum circuit consists of alternating layers of circuits that either apply the mixing or the problem Hamiltonian, a strategy is just a method for generating a list of floats of even length.

##### Random or fixed parameter choice

The strategy `random_or_fixed` generates the list of floating values based on a list that has either floats or the string "rand" as entries.
If the entry is a float, it will be kept as it is. If it is the string `"rand"`, it will be replaced by a random value in the neighbourhood of zero.

|  parameter name       |  description                                                                                         |  type
| --------------------- | ---------------------------------------------------------------------------------------------------- | -----------------------------------------
|  initial_guess        |  a list of values used in choosing inital angles of the various layers of the quantum circuit        |  a list with floats or the string "rand" as entries
|  range                |  the upper limit of the absolute value of a chosen random value                                      |  float
|  repetitions          |  number of independent QAOA runs that are started                                                    |  integer 
  
The lenght of the list `inital_guess` also implicitly sets the number of layers in the quantum circuit.

The following JSON-object is an example for the a configuaration of QAOA using random initialization of all angles in the intervall $(-3,3) \subset \mathbb{R}$:

```
{
    "backend_config": {
        shots: 200,
        simulate: "True",
        noise: "False",
        simulator: "aer_simulator",
        max_iter: 15,
        repetitions: 8,
        classical_optimizer: "COBYLA",
        strategy: "random_or_fixed",
        initial_guess: ["rand", "rand"],
        range: 3
    }
}
```

##### Grid Search

|  initial_guess        |  a list of values used in choosing inital angles of the various layers of the quantum circuit        |  a list with floats or the string "rand" 

The strategy `grid_search` initializes the angles from a grid of parameter values. For each layer of the quantum circuit, a lower and upper bound is given, aswell as the number of points to use.
Then, the algorithm goes over all combination of points and uses them as inital angles for the layers.

The only paremeter needed ist `inital_guess` which is a list of dictionaries, each specifying the grid for their respecitve layer of the circuit. The following table explains the entries these 
dictionaries have:

|  key             |   description                                                                                             |   type   |
| ---------------- | --------------------------------------------------------------------------------------------------------- | -------- |
|  lower_bound     |   the smallest float used as an initial angle                                                             |  float   |
|  upper_bound     |   the biggest float used as an inital angle                                                               |  float   |
|  num_gridpoints  |   The number of floats used an initial angles in the intervall between the lower bound and upper bound    |  float   |

The following JSON-object is an example for a configuration of the strategy `grid_search`:

```
{
    "backend_config": {
        shots: 200,
        simulate: "True",
        noise: "False",
        simulator: "aer_simulator",
        max_iter: 15,
        classical_optimizer: "COBYLA",
        strategy: "grid_search",
        initial_guess: [
            {
                lower_bound: -2,
                upper_bound: 2,
                num_gridpoints: 3
            },
            {
                lower_bound: -2,
                upper_bound: 2,
                num_gridpoints: 3
            }
        ]
    }
}
```

This would start nine QAOA runs witha quantum circuit consisting of one layer with the initial angles being all combinations of the numbers $-2,0,2$.

#### Mixed Integer Linear Programming

The [mixed integer linear programming approach](https://en.wikipedia.org/wiki/Linear_programming) uses PyPSA's native method to cast the problem into a mixed integer linear program and solve it using GLPK. Because this is just a wrapper for PyPSA's native solving method, this has just one option for the maximal duration of the optimization.

|   configuration key  |  impact            
| -------------------- | ------------------ 
| timeout              | time in seconds after which the optimization will be stopped and the current best result will be returned

An example for configuring this solver is very short. Due to the nature of the problem, it is easy to find a decent feasible solution fast, but finding a close to optimal solutions takes a lot of times for bigger problems.

```
{
    "backend_config": {
        "timeout": 10
    }
}
```

### API-token

Using D-Wave's quantum hardware or using an IBM quantum computer for QAOA requires an API token. For D-Wave services, you have to use the key `dwave_API_token` and for IBM the key `IMBQ_API_token` to pass the token.

```
"API_token": {
    "IBMQ_API_token": "Your IMBQ API token",
    "dwave_API_token": "Your D-Wave API token"
}
```

----

## Output format

After a succesful calculation, this service returns a JSON object. This JSON-object has a field for storing all relevant result details. Is also saves the various parameters used to generate the result in order to give some context to the result and fills in default values that were used. The returned JSON has the following fields:

- `results`: this contains information about the solution
- `network`: the serialized network which's unit commitment problem was solved
- `config`: the various configuration parameters that were passed to the solver
- `start_time`, `end_time`: the start and end time in `yyyy-mm-dd-hh-mm-ss` format

For every solver the values in table below are in the `results` object. The table explains the result data that all solvers return. Each solver also adds extra entries that describe aspects that are unique to their solving method. Those won't be explained here.

|  keyword                   |     description  
| -------------------------- | ---------------
| total_cost                 |  Cost of the QUBO representing the unit commitment problem including constants.
| kirchhoff_cost             |  sum of all squared deviations from the kirchhoff constraint
| power_imbalance            |  sum of all linear absolute deviations from the kirchhoff constraint
| total_power                |  total power generated by the solution
| marginal_cost              |  total marginal costs incurred by the solution
| individual_kirchhoff_cost  |  JSON object with stringified `f"({bus}, {snapshot})"` tuples as keys and the deviation from the kirchhoff constraint at that bus and snapshot as values
| unit_commitment            |  JSON object with stringified `f"({generator}, {snapshot})"` tupels as keys and the percentange of the maximal power output of that generator in the solution at that snapshot as values
| powerflow                  |  JSON object with stringified `f"({line}, {snapshot})"` tupels as keys and the powerflow sign through that line at that snapshot as values

In essence, `total_cost` describes how well the QUBO was solved, `marginal_cost` describes the objective function of the unit commitment problem and `unit_commitment` in conjuction with `powerflow` describes the state of the network that has been calculated as the solution.
