The goal of this use case is finding the optimal energy dispatch and transmission flow over time in an energy grid. This helps energy producers and transmission system operators to operate profitably while balancing power generation and demand at all times. The push for decentralized energy sources and demands (e.g. individual photovoltaics, e-mobility) as well as ambitious European CO~2~ emission caps will likely further increase the demand for optimal or near-optimal energy grid costs.

A generic energy system model to model the above mentioned aspects consists of various cost contributions that can be arbitrarily detailed and extended depending on the need of the target situation. In this use case we focus on optimizing the marginal costs associated with running the grid (running costs of generators, transmission lines etc.), capital costs associated with network expansion (installing new generators, transmission lines, ...), and the costs for the start-up/shut-down of generators over time. This pared-down version inherits much of the complexity seen in the full-scale optimization problem, namely, a large number of variables and constraints. The problem can also be extended in numerous ways.
Mathematically, we cast the optimization problem as a mixed integer linear program with two sub-problems: (1) network expansion and (2) the unit commitment problem. Our strategy for solving the mixed integer linear program is to alternate between optimizing these sub-problems for a number of iterations. The complexity of optimizing the marginal costs comes from the large number of variables and constraints (typically on the order of 10-100 million). The sub-problem of optimizing start-up and shutdown costs of generators constitutes the so-called unit commitment problem. Our formulation of unit commitment introduces binary variables that indicate whether a generator is switched on or off for each generator and each time step. Formally the unit commitment problem is NP-hard so we cannot expect a (classical or quantum) computer to solve it in polynomial time. However, the problem is amendable to quantum-inspired solvers and, potentially, to quantum annealers, which might lead to a runtime speed-up.

An energy grid is a graph that models energy generation, storage, demand and transmission. Typically, each node comprises several modes of energy generation, such as coal-fire power plants, solar energy farms, and energy utilization, such as households and industry. A node is a sufficiently localized entity such as a city or a group of cities. Transmission is modelled as a simple flow between nodes connected by power lines. We do not consider the details of impedances in AC/DC components, i.e. the flow is assumed linear. Figure 1 illustrates an energy grid with five nodes.
::: hljs-center

![Figure 1: Graph of an energy grid with five nodes and links between nodes](87905ee7-e493-4e48-a61f-2cfd9cd54903)
*Figure 1: Graph of an energy grid with five nodes and links between nodes*
:::

We want to minimize the annual system costs (in EUR/a) of an energy grid over a long time horizon. The system costs consist of capital cost for the energy generation (annual investment costs in EUR/MWH) and energy transmission as well as marginal costs for the dispatch (in EUR/MWH). The following table provides an overview of all variables in the optimization problem.

|**Variable**    |**Description**|
|      -         |      -        |
|$c_{n, s}$      |Capital costs (in MW) per node n and power plant unit or storage unit (installed power capacity or storage capacity)|
|$c_{l}$         |Capital cost per interconnector $l$|
|$\bar{g}_{n, s}$|Maximum installed generation power of power plant unit|
|$\bar{h}_{n, s}$|Maximum installed storage capacity|
|$F_{l}$         |Maximum transmission capacity per interconnector|
|$o_{n, s, t}$   |Marginal costs per node $n$ and additional used capacity or storage unit capacity|
|$g_{n, s, t}$   |Generation power per node $n$, power plant unit $s$, and time step $t$|
|$h_{n, s, t}$   |Storage capacity per node $n$, storage capacity $s$, and time step $t$|
|$su_{n, s, t}$  |Start-up costs per node $n$, power plant unit $s$, and time step $t$|
|$sd_{n, s, t}$  |Shut down costs per node $n$, power plant unit $s$, and time step $t$|
*Table 1: Overview of variables used in the problem formulation*

Using these variables, we can give a mathematical formulation of the function to be minimized:
$$ \underbrace{\sum\limits_{n, s} c_{n, s} \bar{g}_{n, s} + \sum\limits_{n, s} c_{n, s} \bar{h}_{n, s} + \sum\limits_{l} c_{l} F_{l} }_{\text{capital costs}}
+ \underbrace{\sum\limits_{n, s, t} o_{n, s, t} g_{n, s, t} + \sum\limits_{n, s, t} o_{n, s, t} h_{n, s, t} }_{\text{marginal costs}}
+ \underbrace{ \sum\limits_{n, s, t} su_{n, s, t} + sd_{n, s, t}  }_{\text{shutdown/start-up costs}}.$$

The terms in the first sums correspond to capital costs incurred by the generators, storage units, and transmission lines. Then in the second batch, we add terms that correspond to the marginal costs of the generators and storage units. At last, we add terms that correspond to the power plant start-up or shut-down costs. In a first approximation, capital costs relate to a time-independent maximally installed capacity of the generation unit or storage unit $\bar{g}_{n, s}$ and $\bar{h}_{n, s}$. The marginal costs $o_{n, s, t}$ relate to an actual power plant dispatch with an associated generation power $g_{n, s, t}$ and storage usage $h_{n, s, t}$. 

Currently our data sets do not contain start-up costs $su_{n, s, t}$ or shutdown costs $sd_{n, s, t}$, which reduces the problem complexity. To further reduce the problem size we, in a first step, neglect the storage capacities, thus resulting in the following expression for the cost function:
 $$ \sum\limits_{n, s} c_{n, s} \bar{g}_{n, s} + \sum\limits_{l} c_{l} F_{l} + \sum\limits_{n, s, t} o_{n, s, t} g_{n, s, t} $$

We can translate this into a Unit Commitment Problem, which can then be modeled as a Quadratic Unconstrained Binary Optimization (QUBO) problem. Using Quantum Annealing, Simulated Quantum Annealing (SQA), or Quantum Approximate Optimization Algorithm (QAOA) we can solve this problem and give a suggestion which generators should be used, to match the current network load.
 
The Unit Commitment Problem currently works minimizing the marginal costs, while considering the capital costs as constants, which will be included in later. Thus only the last term of the expression, i.e. $\sum\limits_{n, s, t} o_{n, s, t} g_{n, s, t}$, will be minimized.

# Additional information not available on the platform

Even though the goal is to minimize the marginal costs, various constraints have to be considered to simulate a real problem. Constraints can be the minimum up/down time of generators, the gradual ramp-up and -down times as well as adhering to the Kirchhoff laws. The last one is the most important one, which is why we implemented it first. 

# Kirchhoff constraints
The Kirchhoff law basically states, that at every node in a circuit the energy which enters it has to be equal to the energy leaving it. Thus there cannot be a positive, nor negative energy balance at a node. In terms of optimization, this can be described by the following expression for one node:
$$ \text{minimize: } \left| d + \sum\limits_{i=1}^{m} g_i G_i + \sum\limits_{i=m+1}^n l_i L_i \right| $$
where $d$ is the demand at the node. The second term describes the power generated by all generators and the last term the ammount of energy added to or substracted from the node via transmission lines. The capitalized letter $G_i$ and $L_i$ are binary variables and encode the status of the i^th^ component, being a generator or transmission line, (i.e. on (1) or off (0)) and the lower case letters $g_i$ and $l_i$ describe their generation power and transmission line capacity. $m$ is the number of generators at this node, whereas $n - m$ the number of transmission lines. 
The absolute value in the above expression can as well be interpreted as a square, which will give us the following expression after, summarizing $\sum\limits_{i=1}^{m} g_i G_i + \sum\limits_{i=m+1}^n l_i L_i$ to $\sum\limits_{i=1}^{n} x_i X_i$ and separating the terms:
$$ \text{minimize: } d^2 + 2d\sum\limits_{i=1}^{n} x_i X_i + {\left(\sum\limits_{i=1}^n x_i X_i\right)}^2 $$
Since $d$ is a constant, at one point in time, we can use the quadratic polynomial as our QUBO, thus receiving:
$$ 2d \cdot \sum\limits_{i = 1}^n x_i X_i^2 + \left( \sum\limits_{i=1}^n x_i X_i  \right)^2 \in \mathbb{R}[X_1,\ldots,X_n] $$
Here $X_i$ in the first term has to be squared, to adhere to the QUBO format. However, this can be done without a problem, since $X_i$, just as $G_i$ and $L_i$ before, is a binary variable.

# Marginal costs
So far we have formulated a QUBO, which's minimization is equivalent to satisfying the Kirchhoff constraint. The goal of the optimization problem, though, is minimizing the cost of the generators and fulfill the kirchhoff constraint at every bus. Thus we have to find a QUBO problem which is equivalent to minimizing the marginal costs. We can then obtain a QUBO formulation of the full problem by summing up both QUBOs. At best, an optimal solution of the summation minimizes both parts and thus fulfills the kirchhoff constraint, and minimizes the incurred marginal costs. However, in general it is not possible to simultaneously solve both problems optimally. This requires us to scale the problems, with the difference in scale between both subproblems describing how much penalty in one problem is worth incurring to obtain a better result in the other one. Since the kirchhoff constraint is a mandatory constraint, it’s scale has to be large enough that it is not optimal to violate the kirchhoff constraint in order to use more efficient generators.

The most straightforward way to encode the marginal costs in a QUBO is by adding them to the QUBO as coefficients of the variables describing the status of the generators. Thus, using this approach, we obtain a QUBO defined by:
$$ \sum\limits_{i=1}^{m} g_i c_i G_i^2 $$
where $G_i$ is again the binary variable encoding the status of the i^th^ generator with it's generation power $g_i$. Already $c_i$ are the marginal costs per unit power generated of generator $i$. 
One issue with this formulation stems from the magnitude and variance in generator costs. Because the cost of the generator is given as the product of the power generation and the cost of producing one unit of power, small differences in either of those get scaled up by the other. Because the kirchhoff constraint is dependent on just the power generation, the marginal cost's absolute value will be much higher. 
Since committing generators only incurs costs, the optimal solution for just the QUBO that describes the marginal cost is to commit no generators at all. This solution incurs a high kirchhoff cost. In essence, it is impossible to solve both QUBO's well. Thus we have to significantly scale up the QUBO of the kirchhoff constraint or significantly scale down the QUBO of the marginal costs, in order for the optimization of the kirchhoff constraint to take precedence. We can remedy this a bit by introducing an offset. Because the total power an optimal solution has to produce is constant, we can reduce the absolute values of the marginal costs by introducing a constant offset. Such an offset
doesn’t change the optimal solution of the combined problem but it brings the optimal solution of the marginal costs QUBO closer to the optimal solution of the kirchhoff constraint. We can immediately apply this to the QUBO formulation of the marginal costs to obtain the following expression:
$$ \sum\limits_{i=1}^{m} g_i (c_i - C) G_i^2 $$
where $C$ is a chosen constant offset. Depending on it, the resulting encoding is more or less good-natured with regards to the kirchhoff constraint. However, though mitigating the problem, it doesn't solve it and we still have to use a big difference in scale to get feasible solutions, which^makes it harder to solve. 

Another, still persisting, problem of this encoding is that it doesn't take advantage of the topology of the network, there is no direct interaction between the variables. Any interactions between QUBO variables occur using the interactions of the kirchhoff constraint. In order to amend those problems, we have to use a different approach to model the marginal costs. The core of the issue is that the goal of satisfying the kirchhoff constraint runs contrary to minimizing the cost. The central idea to remedy this is to slightly alter the optimization goal of the marginal cost. Instead of trying to minimize the marginal cost, we try to minimize the squared distance of the incurred marginal cost to a marginal cost goal. 
There are multiple reasons for introducing a marginal cost goal as a new parameter to the problem. We can use this parameter to adjust the marginal cost QUBO and reduce the conflict between it and the kirchhoff constraint. Unlike scaling one subproblem, this doesn’t change the magnitude of the energy contributions of one subproblem. However, this moves the problem of finding a good ratio between both supbroblems to estimating a good marginal cost goal. Another reason for using a squared-distance approach is, that it yielded great results when we applied it to the kirchhoff constraint. It is also more intuitive in the sense, that the absolute values of the marginal costs are irrelevant, but only the relative difference matters. Minimizing the distance to an estimated value sets all values into relation to the estimation as the minimal point. It also implicitly takes the topology of the network into account because the optimal marginal cost that we try to estimate depends on that topology. The better the estimation of the marginal cost, the better the topology of the network is taken into account. It also gives us more information about the quality of the solution. Because we know the estimated costs, we can calculate how well the marginal cost subproblem was solved by comparing the estimation with the cost of the solution. Later, we will use this to propose an iterative approach to update the estimation until we come close enough to the optimal solution.
The straightforward way to encode the marginal costs is to estimate some number $e \in \R$ and repeat the same derivation we used to obtain the QUBO of the Kirchhoff constraint. We would have to substitue the power of a generator with the total cost incurred when committing that generator. The estimation fulfills the role of the load and we pretend that all generators are connected to a global bus. This would lead us to the following QUBO:
$$ \left(-e + \sum\limits_{i=1}^{m} g_i c_i G_i \right)^2 $$
with $G_i$ being a binary variable, this is equivalent to:
$$ 2 \cdot \left(-e + \sum\limits_{i=1}^{m} g_i c_i G_i^2 \right) + \left( \sum\limits_{i=1}^{m} g_i c_i G_i \right)^2 $$
This encoding can be split up into two hamiltonian matrices: The first one is for encoding relative contributions to the marginal cost which is estimation indepedent. The other one is a diagnoal matrix and contains the marginal cost that we try to match. 
In another approach we can improve this formulation by offsetting the marginal cost. This leads us to:
$$ \left( \sum\limits_{i=1}^{m} g_i (c_i - C) G_i \right)^2 $$
which is equivalent to:
$$ 2 \cdot \left(-P \cdot C \cdot \sum\limits_{i=1}^{m} g_i c_i G_i^2 \right) + \left( \sum\limits_{i=1}^{m} g_i c_i G_i \right)^2 $$
where $P$ is the total load of the network and thus equivalent to $\sum\limits_{i=1}^{m} g_i G_i$. By using the offset, we spread the contribution on the diagonal of the hamilitonian matrix across the different second-order interactions. Doing this and thus centering the marginal cost around zero has the benefit that there is no linear bias wether generators are committed or not, if we ommit the Kirchhoff constraint, which was one of the weak points of the direct encoding.

# Combined QUBO
Combining the QUBO's discussed until now will be highly dependant on the estimation of the marginal costs, as well as, the linear factors scaling the two QUBO's. The combined QUBO has thus three parameters $a_1, a_2, e \in \R$, that we can adjust: a linear factor for each, the marginal cost and for the kirchhoff constraint, contribution, and the estimation of the marginal costs.
$$ QUBO_{total} = a_1 \cdot QUBO_{Kirchhoff} + a_2 \cdot QUBO_{Marginal} $$
Both QUBO's describe the squared-distance of some generator configuration with regards to some metric (kirchhoff, marginal cost). Because of that, the cost grows quadratically with the distance of the solution to the respective target value. This means that regardless of the chosen scaling factors $a_1$ & $a_2$, we will come to a point at which improving one summand improves the overall solution even if it incurrs a penalty in the other one. However, the range of this depends on the ratio of the scaling factors.
Let’s assume that the $QUBO_{Kirchhoff}$ only has integer valued energies. Then, an optimal solution has kirchhoff cost 0 and the next best solution has cost 1. In order for the solution of the combined QUBO to incur a kirchhoff cost of 1, the difference in marginal cost has to be at least $a_2/a_1$. Thus, the solution of the combined QUBO is the optimal solution of the unit commitment problem if for the marginal cost $e_{opt}$ of the optimal solution the equation:
$$ |e_{opt} − e| < a_2/a_1 $$
holds.