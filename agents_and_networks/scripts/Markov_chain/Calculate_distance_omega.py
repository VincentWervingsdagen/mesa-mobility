import pandas as pd
from pyproj import Transformer
import os
import construct_markov_chains as MC

script_dir = os.path.dirname(os.path.abspath(__file__))

cell_file=os.path.join(script_dir, '..','..', 'data', '20191202131001.csv')
bounding_box = (4.1874, 51.8280, 4.593, 52.0890)
antenna_type='LTE'
level = 'antenna'

states = MC.state_space_Omega(cell_file, bounding_box, antenna_type,level)
distance_matrix = MC.distance_prior(len(states),states,level=level)
print(distance_matrix)
distance_matrix.to_csv('./LTE/postal.csv')