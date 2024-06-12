import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Bounding box for trajectory simulation
BOUNDING_BOX = (4.3480,52.0036,4.3741,52.0185)
# Bounding box increase for connecting cell tower simulation
BOUNDING_INCREASE = 0.01

# Start and end date of simulation
START_DATE = '2023-05-01'
END_DATE = '2023-06-10'

# Cell tower location file and pickled coverage model
CELL_FILE = os.path.join(script_dir, 'data', '20191202131001.csv')
COVERAGE_FILE = os.path.join(script_dir, 'data', 'coverage_model')

# Locations for output trajectory and cell tower connections
OUTPUT_TRAJECTORY_FILE = os.path.join(script_dir, 'outputs','trajectories','output_trajectory.csv')
OUTPUT_CELL_FILE = os.path.join(script_dir,'outputs','trajectories','output_cell.csv')

# Building file and street file
# Download regions from following location https://download.geofabrik.de/europe/netherlands.html
BUILDING_FILE = os.path.join(script_dir,'data','zuid-holland','gis_osm_buildings_a_free_1.zip')
STREET_FILE = os.path.join(script_dir,'data','zuid-holland','gis_osm_roads_free_1.zip')
