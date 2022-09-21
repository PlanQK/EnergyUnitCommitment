The goal of an energy provider is to provide energy to consumers. These range from private households to industrial factories. For now, we focus on electrical power since
other utilities like gas have their own special properties.
In order to meet the demand of the consumers, the provider has various types of power generators at their disposal, each with their own advantages and disadvantages. For example
solar power is cheap to produce, but it is not always available while coal-fire power plants are more expensive to operate but are not dependent on external circumstances to produce power.
This power also has to be delivered to the consumer, using a vast network of transmission lines which are used to transport power across regions.
All of these generators, transmission lines, consumer demand and power storage facilities make up the energy grid.

The problem an energy provider has to solve is now clear: They have to find a optimal energy dispatch and transmission flow which meets the demand of the consumers while keeping costs low.
The push for decentralized energy sources and demands (e.g. individual photovoltaics, e-mobility) as well as ambitious European CO~2~ emission caps will likely further increase the complexity of this problem and thus the demand for optimal or near-optimal solutions.

A generic energy system model to model the above mentioned aspects consists of various cost contributions that can be arbitrarily detailed and extended depending on the need of the target situation. In this use case we focus on optimizing the marginal costs associated with running the grid (running costs of generators, transmission lines etc.) and capital costs associated with network expansion (installing new generators, transmission lines, ...).
One key aspect that we want to model is that in general power generators can't be turned on or off on a whim. 
Instead the start-up and shut-down procedures incur addiional costs and some types generators have a minimal up and downtime. The combinatorial problem of choosing which generators
to turn on and which to turn off is called the unit commitment problem. 

This simplified down version inherits much of the complexity seen in the full-scale optimization problem, namely, a large number of variables and constraints. 

This pared-down version inherits much of the complexity seen in the full-scale optimization problem, namely, a large number of variables and constraints. The problem can also be extended in numerous ways.
Mathematically, we cast the optimization problem as a mixed integer linear program with two sub-problems: (1) network expansion and (2) the unit commitment problem. Our strategy for solving the mixed integer linear program is to alternate between optimizing these sub-problems for a number of iterations. The complexity of optimizing the marginal costs comes from the large number of variables and constraints (typically on the order of 10-100 million). The sub-problem of optimizing start-up and shutdown costs of generators constitutes the so-called unit commitment problem. Our formulation of unit commitment introduces binary variables that indicate whether a generator is switched on or off for each generator and each time step. Formally the unit commitment problem is NP-hard so we cannot expect a (classical or quantum) computer to solve it in polynomial time. However, the problem is amendable to quantum-inspired solvers and, potentially, to quantum annealers, which might lead to a runtime speed-up.


An energy grid is a graph that models energy generation, storage, demand and transmission




