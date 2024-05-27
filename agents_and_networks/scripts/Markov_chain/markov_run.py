from markov_model import MarkovChain

import os

script_dir = os.path.dirname(os.path.abspath(__file__))

MarkovChain(cell_file=os.path.join(script_dir, '..','..', 'data', '20191202131001.csv'),
            observation_file = os.path.join(script_dir, '..','..', 'outputs','trajectories','output_cell.csv'),
            state_space='Omega',#options are either Omega or observations. Omega is quite slow already with antenna/postal.
            bounding_box=(4.1874, 51.8280, 4.593, 52.0890),
            state_space_level='postal', #options are either antenna, postal, postal3.
            prior_type='uniform', #options are either uniform, distance, population.
            markov_type='discrete', #options are either discrete or continuous.
            distance ='cut-distance', # options are either cut-distance or freq-distance.
            loops_allowed = True # Decide whether you want to have self loops in your markov chain or not.
            )