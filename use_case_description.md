The goal is to efficiently meet the power demand in an electrical grid using installed power generators, which have various constraints on their availability and flexibility.

# Introduction

The European power grid provides power to millions of households and industries in various forms like electricity, heat and gas. In the case of the electrical power grid, so called transmission system operators (TSO) are tasked with running the grid of their respective countries. They provide electricity market players with access to it and ensure the safe operation and maintenance of the system.


One aspect of this is to balance the power generation and demand at all times to avoid brown- or even blackouts. It is in the TSO's interest to achieve this while minimizing costs that are incurred for running and expanding the grid. Expanding the grid consists of building new power plants, transmission lines, power storage facilities or expanding the capabilities of already existing network components Running the grid incurs costs due to maintenance costs or fuel that is required by some types of power plants. In this use case, we want to focus on the latter optimization problem for now and ignore the problem of optimizing the cost of network expansion.

This requires a model of the energy grid in order to formulate the exact optimization problem and level of detail that we want to solve.


# The Unit Commitment Problem


A generic energy system model to represent an energy grid consists of various cost contributions and constraints that can be arbitrarily detailed and extended depending on the need of the target situation. More accurate models require more constraints and describe the costs in more detail which makes the problem harder to solve. We want to focus on a simplified model with one central idea to model power generators, which leads us to the so called unit commitment problem.

An energy grid is a graph that models energy generation, storage, demand and transmission. Typically, each node comprises several modes of energy generation, such as coal-fire power plants, solar energy farms, and energy utilization, such as households and industry. A node is a sufficiently localized entity such as a city or a group of cities. Transmission is modelled as a simple flow between nodes connected by power lines. We do not consider the details of impedances in AC/DC components, i.e. the flow is assumed linear. Figure 1 illustrates an energy grid with five nodes.
::: hljs-center

![Figure 1: Graph of an energy grid with five nodes and links between nodes](87905ee7-e493-4e48-a61f-2cfd9cd54903)
*Figure 1: Graph of an energy grid with five nodes and links between nodes*
:::

The power flow and individual power generation is modeled over a period of time with discrete time steps. The resolution of the time steps depends on how accurately you want to model the system. For now, we assume that each time step represents an hour of passed time.

The main aspect that we want to model is that power generators cannot provide arbitrary amounts of power within their generating capacity. Instead we distinguish between generators that are turned on and those that are turned off. Various constraints depend on this state of the generators. For example, after turning on a coal power plant, it has to run for a minimal amount of time before it can be safely turned off. In general, the constraints impose restrictions on the state of generators across multiple time steps. Choosing which generators to commit at each time step constitutes the Unit Commitment Problem.

Due to the combinatorial nature of the problem, it is NP-hard. This makes it into an excellent candidate for the exploration of quantum computing because the two states of generators naturally correspond to the state of qubits. The problem being hard to solve also means that quantum methods might provide significant gains in both speed and quality of solutions.


In order to keep the problem relatively simple, we are only going to consider a handful of constraints that pertain the state of the generators. The following table describes which constraints and which costs we are going to consider in the further optimization. Each row either describes a cost, which has to be minimized, or a constraint that has to be satisfied by a solution.

|**Aspect**    |**Description**|   **Type** |
|      -                   |      -        |  - |
| Marginal cost            |  Each generator has a specific price for generating a MW of power |  cost  |
| Kirchhoff's Law          |  At each node and timestep, the generated power and demand has to be balanced |  constraint  |
| Startup/Shutdown cost    |  Startup- and Shutdown procedures incur additional costs on top of the costs for producing power |  cost  |
| Minimal Up- and Downtime |  After turning a generator on or off, it can't be turned off or on again until a generator specific amount of time has elapsed |  constraint  |
| Minimal Poweroutput      |  If a generator is turned on, it has to produce a minimal (non-zero) amount of power |  constraint  |
| Rampup and Rampdown      |  When a generators is turned on or off, the poweroutput slowly changes from no power to full power and vice versa |  constraint  |


## The classical solution

The unit commitment problem can be solved by casting it as a mixed integer linear program (MILP). For each time step the output of the generators and the powerflow of  transmission lines are modeled as real-valued variables and the state of a generator is given by a binary variable. The constraints in the table above then can all be encoded using different linear equations. Commercial solvers like CPLEX or Gurobi can then used to solve the ensuing linear program.


## Preparing the model for a quantum Computer


However, we cannot reuse the the MILP - formulation to run the problem on a quantum computer. Instead, we have to turn to another kind of mathematical optimization problem which is called a quadratic unconstrained binary optimization (QUBO) problem. These kind problems can easily be loaded into a quantum computer using the API of the respective provider. Choosing the encoding of the network components and formulation of problem constraints as a QUBO has a significant impact on the quality of the solution. Thus we have explored how different ideas for encoding the same constraint as a QUBO influences the results.

An additional problem that has to be adressed is that the current state of the hardware imposes restrictions on the problems that can be solved. The everpresent noise of current quantum computers also limits the quality of any solution. Due to the hardware size, some quantum methods can only handle a handful of qubits (QAOA) or a couple of thousands qubits (Quantum Annealing). The limited connectivity of qubits further constraints which problems can be represented and solved on quantum hardware.

In order to get around this limitations, we have also turned to simulations of quantum computing. In particular, a Monte Carlo simulation of Quantum Annealing isn't restricted by any noise, hardware size or connectivity, which makes it a good candidate to evaluate QUBO-models independent of current quantum hardware limitations.

We have elected to limit ourselves to solving (randomly generated) toy problems. On the one hand, real-world problems are so huge, that it would be incredibly computationally expensive to use them for studying. On the other hand, even smaller problems capture the complexity of the problem and can be unsolvable by classical methods in a reasonable amount of time.

