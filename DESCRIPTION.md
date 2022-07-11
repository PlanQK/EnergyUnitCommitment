## Energy Unit Commitment
This service contains various solvers for optimizing a simplified version of the unit commitment problem of an energy grid.  The unit commitment problem consists of an energy grid consisting of power generators, transmission lines and loads.  The problem is to optimally choose which generators to commit such that all loads are met and the generators incur minimal costs.  You can find more information on the problem [here](https://platform.planqk.de/use-cases/e8664221-933b-4410-9880-80a6900c9f86/).

### input format
The input of this service has to be in JSON-format. This input has the two fields `data` and `params` to specifiy the network and which solver to use respectively. For the field `data`, which contains the network, this service requires it to be a serialized [PyPSA network](https://pypsa.readthedocs.io/en/latest/). You can obtain a serialized PyPSA network by first exporting it to an xarray using [export_to_netcdf](https://pypsa.readthedocs.io/en/latest/api_reference.html#pypsa.Network.export_to_netcdf). Calling the resulting xarray's method `to_dict()` returns a dictionary which can be saved in JSON-format.

In order to choose the solver and configure it's parameters, the field `params` is used to pass another JSON-object. Because there are multiple solvers and each solvers has some unique parameters, there are a lot of possible options to specify in that field. Therefore we will go through the structure of the JSON object and explain how you can configure the various aspects of the different solvers.

The JSON-object contains these four fields: `backend`, `backend_config`, `ising_interface` and `API_token`. They specify which solver is going to be used and it's configuration. The field `backend` contains a string denoting which solver to use and `backend_config` contains configuration parameters of that solver. Most of the solvers require the unit commitment problem to be cast as a [quadratic unconstrained binary optimization problem (QUBO)](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization). If a solver requires a QUBO formulation of the unit commitment problem, the field `ising_interface` specifies all relevant details on how the QUBO is built.
At last, the field `API_token` contains a JSON-object that can be used to pass API tokens of services which are used by solvers that access quantum hardware.

## Service Configuration
This service supports various solvers for obtaining solutions of the unit commitment problem.  Currently, the main solvers that can be used consist of:
- Mixed Integer Linear Programming (GLPK)
- Simulated Quantum Annealing
- Quantum Annealing

The following solvers can also be chosen for solving the unit commitment problem.  Currently, you can't pass any parameters to them, which makes them unsuited for solving larger problems.

- Tabu search
- steepest descent
- D-Wave's hybrid QUBO solver

In principle, this service also contains a solver using the [quantum approximation optimization algorithm](https://qiskit.org/textbook/ch-applications/qaoa.html), but it is currently unavailable due to a bug. You can also choose simulated annealing as the solver, but it is implemented as simulated quantum annealing with a vanishing transverse field so you can just use simulated quantum annealing instead.

The following table lists all supported solvers and how to choose them by passing the correct string to the `backend` field. Each solver also supports an additional configuration keyword which can also be used to passed parameters. If both the `backend` and the solver specific configuration field set the same parameter, the latter value takes precedence.

|   solver           |   keyword    |   description                                                                               |  configuration keyword  |  uses QUBO  | API token
| ------------------ | ------------ | ------------------------------------------------------------------------------------------- | ----------------------- | ----------- | ---------
| sqa                | sqa          | performs simulated quantum annealing. Default solver if none is speficied                   |  sqa_backend            |  Yes        | None
| annealing          | classical    | performs (classical) simulated annealing                                                    |  sqa_backend            |  Yes        | None
| tabu search        | dwave-tabu   | performs tabu search as it is in d-waves ocean package                                      |  dwave_backend          |  Yes        | None
| steepest decent    | dwave-greedy | performs steepest descent as it is in d-waves ocean package                                 |  dwave_backend          |  Yes        | None
| hybrid solver      | dwave-hybrid | uses d-waves hybrid solver in the cloud                                                     |  dwave_backend          |  Yes        | d-wave
| quantum annealing  | dwave-qpu    | performs quantum annealing using d-waves quantum annealer                                   |  dwave_backend          |  Yes        | d-wave
| qaoa               | qaoa         | performs QAOA using IBM's qiskit by either simulating or accessing IBM's quantum computer   |  qaoa_backend           |  Yes        | IBMQ
| glpk               | pypsa-glpk   | solves a mixed integer linear program obtained by pypsa using the GLPK solver               |  pypsa_backend          |  No         | None


Since the main focus of this service lies on (simulated) quantum annealing and the MILP reference solution, configuring tabu search and the hybrid algorithm will be added later. Thus, choosing a solver adds the following entries the `params` dictionary:

```
{
    "backend": "{keyword}",
    "{configuration keyword}": {}
}
```

Because all solvers except the mixed integer linear program require a QUBO formulation, we will explain how to configure the QUBO model first.

### Configuring the QUBO-model

In order to specifiy which QUBO an underlying solver uses as a basis to solve the unit commitment problem, you have to specify which methods are used to represent network components as QUBO-variables. For each constraint and the optimization goal, you then have to specify which method is used to construct a QUBO using the aforementioned representation of network components as QUBO-variables. Since the total QUBO is obtained by summing all QUBOs encoding some subproblem of the optimization problem, those can be done independently.

As a rough outline of the underlying model, a QUBO-variable represents a quantized piece of energy with it's value denoting if that energy is available or vanishes. A network component is then represented using groups of QUBO-variables with it's state being described by the various values these QUBO-variables can be assigned to. The sum of the quantized energy represents the energy that that network component has in a particular state.

For each aspect of the the unit commitment, the corresponding value of that QUBO-variable with regard to that aspect can be calculated using the underlying value of the energy. For example, if you want to handle the marginal costs, the value of the QUBO-variable belonging to a group that represents a generator, can be calculated as the product of the  energy it represents and the marginal costs this generator incurs for each unit of energy.

#### QUBO representation of a transmission line

In order to represent transmission lines, the capacity of the line has to be split up into multiple QUBO-variables. We have decided to also impart a direction of the flow onto the QUBO-variables instead of encoding the direction of the flow into an addtional QUBO-variable. This matters for the kirchhoff constraint because the direction of flow in a transmission line determines the sign of the energy at a bus

In order to specify how a line is represented as a collection of QUBO variables, we pass a method that takes the capacity of the line and returns a list of weights which represent pieces of energy. This list of weights has to fulfill the condition, that no sum of entries of that list exceeds the capacity of the transmission line. The weights also have to have signs which represent the direction of flow.  The chosen list of weights determines how finely you can represent a flow of energy. Thus, the sum of all positive weights, as well as the sum of all negative weights, should come close to the (negative) capacity of the transmission line to represent the maximum possible flow.

The following table shows how to configure which transmission line representation to use when constructing the QUBO.  The key for this is `formulation` in the the `ising_interface` field.

|  parameter value |   description of line representation                                                                  | additional information
| ---------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------
| fullsplit        | Each QUBO-variable carries a unit of energy for each direction                                        | This scales badly with line capacity. Has a high runtime, but also higher quality of solution
| customsplit      | a compromise of minimizing the number of variables and minimizing the difference in magnitude         | only accepts integer values between 1 and 4 so far.
| cutpowersoftwo   | uses a binary representation in each direction with a cut off at the highest order                    | cheapest option in QUBO-variables wise, but has huge differences in magnitude of weights

In total, this adds the following entries to the configuration.
```
{
    "ising_interface": {
        "formulation": {parameter value}
    }
}
```

#### QUBO representation of a generator

As of now, generators are encoded with a single QUBO-variable. Thus, they can only have two states: Committed and producing full power or turned off. We will later add being able to represent a range of power, as well as specifying a minimal power output if they are committed

#### Encoding constraints

Since a QUBO is by definition unconstrained, we will not distinguish between constraints of the problem like the kirchhoff constraint or the optimimization goal. When using a QUBO-based solver, various constraints can be represented as a QUBO by using a langrangian function whose minima are exactly those states, that satisfy the constraint. The total QUBO is then simply the weighted sums of all langrangian function if they are all polynomial functions of order two. Therefore, each other field in the `ising_interface` value contains the full information needed to construct the QUBO corresponding to the constraint indicated by the key.

The following table describes which key is used for which constraint of the unit commitment problem. Each of these options support the option `scale_factor` which is the factor used when summing up all QUBO's to obtain the full QUBO.

|  keyword             |  problem constraint description
| -------------------- | ----------------------------------------------------
| kirchhoff            | The kirchhoff constraint requires that the supplied energy is equal to the load at all buses
| marginal_cost        | The marginal costs is to be minimal among all feasible solutions

Any other key that isn't used to describe how network components are encoded or in the above table as a listed constraint will be ignored. Adding or removing any constraint from QUBO can simply be done by ommitting the key that corresponds to that constraint. We will now go over the various constraints, and how we can choose different approaches to encode it into a QUBO. While all these approaches are technically a correct way, they have different limitations and properties.

#### Kirchhoff constraint

The kirchhoff constraint enforces that a for a feasible solution, the load is equal to the supplied energy at each bus. We model this as a QUBO by summing up QUBOs that enforce this constraint for each bus. That QUBO's interactions are set up in a way, that the cost of it is equal to the squared distance of the load at the bus to the supplied power. Because this constraint is integral to the problem, it will always be added to the problem and only has the parameter `scale_factor`. 

For example, adding the kirchhoff constraint with a factor of `2.0` to the QUBO would require adding the following entry to the `ising_interface` :
```
{
    "ising_interface": {
        "kirchhoff": {
            "scale_factor": 2.0
        }
    }
}
```

This constraint also doesn't support any other encoding than the squared distance approach.

#### Marginal Costs

Minimizing the marginal costs is technically not a constraint that a solution has to satisfy, but just the optimization goal.  Because we have to write it as QUBO, it is similar to a constraint and can be treated as such when configuring the solver.

Directly encoding the cost into the energy cost requires huge scaling factors, which in return leads to a bad spectral gap. Therefore, most strategies that we implemented work by estimating the total marginal cost and encoding the squared distance to that estimation. The caveat here is that the QUBO is technically only equivalent to the unit commitment problem if the estimation is "good enough" or the scaling factors are big enough.  Luckily, even solutions of imprecise QUBO models allow us to get closer to the true value of the minimal marginal cost. By iteratively improving our estimation, we can get close enough to the true value that our QUBO is equivalent to the unit commitment problem. This allows us to get around the issue of the direct encoding, which have a bad spectral gap due to requiring huge scaling factors.

Using an estimation based approach also has other benefits. First, the quadratic growth of the squared distance ensures that the constraint will always outscale any linear factors. This prevents the marginal costs to be irrelevant when used in conjuction with a small scaling factor. The other benefit is that good estimations lead to well conditioned problems since it means that both the kirchhoff constraint and the marginal costs can be optimally solved by the same solution which is not the case for the direct encoding. This allows us to choose scaling factors that are relatively similar without the optimization of the marginal costs superceding the optimization of the kirchhoff constraint. The following table describes various strategies to encode the marginal cost into a QUBO. 

|    configuration value              |                 strategy description                                 
| ----------------------------------- | ------------------------------------------------------------------
| marginal_as_penalty                 | direct translation from cost to energy as first order interactions                                  
| global_cost_square                  | squared distance of total cost to a fixed estimation                    
| global_cost_square_with_slack       | squared distance of total cost to an estimation with slack variables
| local_marginal_estimation_distance  | squared distance of total cost at each bus to estimations               

Each strategy supports the keys `scale_factor` and `offset_estimation_factor` whose values are floats.  The former simply scales the QUBO encoding of the marginal cost by a linear factor while the latter introduces an offset into the marginal cost values of the generators based on their relative power output.  It does so by using a reference value and the `offset_estimation_factor` to calculate the total offset of a solution that satisfies the kirchhoff constraint. Using the kirchhoff constraint, that offset can be distributed across all generators. By using the kirchhoff constraint, we can prove that this change in marginal costs doesn't alter the optimal solution of the unit commitment problem.

The reference value used by the `offset_estimation_factor` is the simplest lower bound you can give for the cost of the optimal solution That value is calculated by committing the most efficient generators regardless of their associated bus up to the total load of the problem. This solution matches only the total power necessary, but no solution can be more efficient than the solution using the most efficient generators.

A factor of `0.0` introduces no offset of the marginal costs.  The factor `1.0` introduces a total offset equal to the reference value. This means that any solution to the unit commitment problem that satisfies the kirchhoff constraint has a lower marginal cost equal to the reference value with regards to the offset marginal costs.

In order to reduce differences in magnitudes of different interactions, we have slightly changed how the distance to the estimation is encoded. Instead of adding a parameter that is used to formulate a QUBO that describes the squared distance to that value, we instead assume that the marginal costs of the optimal solution are always zero. We adjust the effective estimation of the marginal costs by using various offsets. The goal is to find an offset of the marginal costs, such that the optimal solution has neglibile distance to zero with regards to the offset marginal costs. The difference to just encoding the squared distance to some value in the same manner that we do it for the kirchhoff constraint, is that it leads to huge values on the diagonal of the corresponding hamiltonian matrix. Our approach allows us to transform first-order interactions into second-order interactions, using the kirchhoff constraint to guarantee that the combined problem have the same solutions.

The option `global_cost_square_with_slack` has three more options on top of the `offset_estimation_factor`.  The idea is to introduce slack variables that act like generators, that are irrelevant to the kirchhoff constraint, but, if active, contribute some fixed value to the marginal cost. Thus it can be used to slightly adjust the estimation during the optimization run. The following table describes the parameters that are used to describe these slack variables.

|  keyword                  |  description 
| ------------------------  | ------------------------------------------------------  
|  formulation              | a string describing how the marginal costs are encoded. The value is one of the configuration values above for the strategy
|  offset_estimation_factor | a float which offsets marginal costs. Higher values lead to lower total marginal costs with regards to the offset
|  slack_type               | a string that specifies by which rule the slack weights are generated. 
|  slack_scale              | additional linear scaling value for the slack variable weights
|  slack_size               | number of slack variables to be generated according to the slack_type rule

The only supported value for `slack_type` is `binary_power`, which configures weights of slack variables as ascending powers of two.  In conjuction with `slack_size`, which specifies how many slack variables are created, the slack in marginal costs can be set up as as any fixed length binary number. The `slack_scale` works similar to `scale_factor`, being a float that scales all slack weights, allowing either bigger or smaller than integer step size.

In total, the JSON-object for configuring the QUBO looks like this.  The following JSON-object can be used as a template, using the above tables to swap out the appropiate values.

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
