## Energy Unit Commitment
This service solves a simplified version of the unit commitment problem of an energy grid. Right now, a lot of additonal constraints on solutions are not supported. 
The unit commitment problem consists of a network containing power generators and loads. The problem is to optimally choose generators to commit/ turn such that all
loads are met and the generators incur minimal costs.

## Service Configuration
This service supports various solvers for obtaining solutions of the unit commitment problem. Right now, the following solving methods are supported:
- Mixed Integer Linear Programming (GLPK)
- Simulated Quantum Annealing
- Quantum Annealing
- Quantum Approximation Optimization Algorithm

### input format
This service takes a serialized [PyPSA network](https://pypsa.readthedocs.io/en/latest/) as it's input and a json for the solver parameters. You can obtain a serialized PyPSA network
by first exporting it to an xarray using [export_to_netcdf](https://pypsa.readthedocs.io/en/latest/api_reference.html#pypsa.Network.export_to_netcdf) and then using `to_dict()` to obtain the json. 

#### Mixed Integer Linear Programming
The [mixed integer linear programming approach](https://en.wikipedia.org/wiki/Linear_programming) uses pypsa's native method to cast the problem into a mixed integer linear program and solve it using a solver.
The service has GLPK installed but you can also return a pyomo model of the problem to run it locally

#### Simulated Quantum Annealing
This solver casts the problem as a [quadratic unconstrained binary optimization problem](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) (QUBO) and solves it uses monte carlo simulation of quantum annealing 
to find an optimal solution. You can find more information on simulated quantum annealing [here](https://platform.planqk.de/algorithms/4ab6ed1f-9f5e-4caf-b0b2-59d1444340d1/) and on quantum annealing [here](https://platform.planqk.de/algorithms/786e1ff5-991e-428d-a538-b8b99bc3d175/)

#### Quantum Annealing
Quantum Annealing uses D-Wave's quantum hardware to perform quantum annealing. This requires an API - token. Due to to hardware limitations, problem size is significantly
limited and solutions aren't as accurate as other solvers.

#### Quantum Approximation Optimization Algorithm (QAOA)
The quantum approximation optimization algorithm build a parametrized quantum circuit to solve the problem. Then, it uses a classical optimizer to adjust the parameters so the
circuit better fits the problem.

