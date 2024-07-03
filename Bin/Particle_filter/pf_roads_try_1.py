from Particle import ParticleFilter
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

ParticleFilter(roadnetwork_file=os.path.join(script_dir, '..','..', 'data', 'zuid-holland', 'gis_osm_roads_free_1.zip'),
                coverage_file_path=os.path.join(script_dir, '..','..', 'data', 'coverage_model'),
                observation_file=os.path.join(script_dir, '..','..', 'outputs', 'trajectories','output_cell.csv'),
                output_file=os.path.join(script_dir, '..', '..', 'outputs', 'trajectories', 'output_cell_pf_roads.csv'),
                walking_allowed=False,
                #bounding_box =(4.3338, 51.9853, 4.3658, 52.0204),  # Delft
                bounding_box=(4.1874, 51.8280, 4.593, 52.0890), #Zuid holland
                N=10,
                method=1 # Use brownian bridge if value is 1. Use 2 if you want a very slow implementation of the particle filter.
                          # Otherwise use a faster, although less realistic particle filter approach.
)

