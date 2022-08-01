"""The `Backends` module contains all solvers for the unit commitment problem. They all
share the base class BackendBase with similar solvers grouped in their own file. Currently,
these categories are:
    - SQA-based solvers,
    - MILP-based solvers,
    - D-Wave solvers
    - QAOA (using IBM's qiskit runtime).

In conjunction with the backends, this module also contains a class for reading the input
and transforming it to call the solver correctly. Finally, for any solvers that requires a quadratic
unconstrained binary optimization formulation, this module provides the IsingBackbone class
which serves as a layer of abstraction between the qubits and the network components. It works
in conjunction with the `IsingSubproblem` interface, which can be used to encode a constraint in QUBO form.

You can extend the networks that can be solved by extending the IsingBackbone class. In order to
add a new constraint, you have a write a class that adheres to the `IsingSubproblem` interface. Then you pass
it as an entry to the `ising_backbone` value of the configuration.
"""

from .SqaBackends import ClassicalBackend, SqaBackend
from .DwaveBackends import (
    DwaveTabu,
    DwaveSteepestDescent,
    DwaveCloudHybrid,
    DwaveCloudDirectQPU,
    DwaveReadQPU,
)
from .PypsaBackends import PypsaGlpk, PypsaFico
from .QaoaBackend import QaoaQiskit
