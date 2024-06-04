import pandas as pd

from markov_model import MarkovChain
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


#Options for running the program.
normal_phone = ['0_1','0_2','1_1','1_2']
burner_phone = ['0_1','0_2','1_1','1_2']
# normal_phone = ['0_1']
# burner_phone = ['1_1']
norm = 'frobenius' # options are either cut_distance,important_cut_distance, freq_distance, frobenius or trace norm.
level = 'postal2' # options are either antenna, postal, postal3,postal2.
state_space = 'observations' # options are either Omega or observations.
prior_type = 'uniform' # options are either zero, uniform, distance, population.

result = pd.DataFrame(index = normal_phone,columns=burner_phone)

for i in normal_phone:
    for j in burner_phone:
        result.loc[i,j] = MarkovChain(
            cell_file=os.path.join(script_dir, '..','..', 'data', '20191202131001.csv'),
            observation_file = os.path.join(script_dir, '..','..', 'outputs','trajectories','output_cell_30days.csv'),
            distance_matrix_file = 'agents_and_networks/scripts/Markov_chain/LTE_dist2/{}.csv'.format(level),
            state_space=state_space,
            bounding_box=(4.1874, 51.8280, 4.593, 52.0890),
            state_space_level=level,
            prior_type=prior_type,
            markov_type='discrete', #options is only discrete.
            distance =norm,
            loops_allowed = True, # Decide whether you want to have self loops in your markov chain or not.
                                      # No self loops make it hard to distinguish H_p pairs,
                                      # because there will be no transitions that are frequent enough.
            phone_normal = i, # Agent number, followed by phone number
            phone_burner = j # Agent number, followed by phone number
            )

def extract_score(obj):
    return obj.get_score()


print('The {} matrix:'.format(norm))
print(result.map(extract_score))