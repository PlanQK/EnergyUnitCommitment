from .ising_backbone import IsingBackbone
from .qubo_transformator import TspTransformator
from .ising_subproblems import HamiltonianPathSubproblem
from ..backends.input_reader import InputReader, GraphReader
#from .pypsa_networks import create_network

backbone = IsingBackbone()
#inputreader= InputReader(network='network_4qubit_2_bus.nc', config='config-all.yaml')
config = {"hamiltonian": {}
              }
graph = {'(1,2)':
  1,
'(2,3)':
  1,
'(3,4)':
  1,
'(4,1)':
  1,
'(2,1)':
  1,
'(3,2)':
  1,
'(4,3)':
  1,
'(1,4)':
  1,
  '(1,3)':
  100,
  '(3,1)':
  100,
  '(2,4)':
  100,
  '(4,2)':
  100,}

tsp = TspTransformator(graph, config)

qubo = tsp.transform_network_to_qubo()