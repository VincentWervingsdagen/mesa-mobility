import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import csv
from tqdm import tqdm

from pyproj import Proj, Transformer
from datetime import datetime, timedelta
from telcell.data.models import Measurement, Point, RDPoint
from random import choices
from config import BOUNDING_BOX, BOUNDING_INCREASE, START_DATE, END_DATE, OUTPUT_TRAJECTORY_FILE, OUTPUT_CELL_FILE, \
    CELL_FILE, COVERAGE_FILE

import os

script_dir = os.path.dirname(os.path.abspath(__file__))

"""
Script to obtain the cell tower samplings from a pre-existing coverage model
"""


def main(model_params):
    # Retrieve start date
    start = datetime.strptime(model_params["start_date"], "%Y-%m-%d")

    # Expand bounding box size
    increase = BOUNDING_INCREASE

    # Setup output file
    output_file = open(model_params["output_file"], 'w')
    output_writer = csv.writer(output_file)
    output_writer.writerow(
        ['id', 'owner', 'device', 'timestamp', 'cellinfo.wgs84.lat', 'cellinfo.wgs84.lon', 'cellinfo.azimuth_degrees',
         'cellinfo.id', 'cellinfo.postal_code'])

    # Read in cell towers:
    df_cell = pd.read_csv(model_params["cell_file"])

    # As coverage model does not take certain information into account (such as safe distance and power)
    # we remove this information and drop any duplicates.
    df_cell = df_cell.drop(['Samenvatting', 'Vermogen', 'Frequentie', 'Veilige afstand', 'id'], axis=1)
    df_cell = df_cell.drop_duplicates()

    # Only consider LTE (4G) for now
    df_cell = df_cell.loc[df_cell['HOOFDSOORT'] == "LTE"]

    # Transform to wgs84
    df_cell['lat'], df_cell['lon'] = Transformer.from_crs("EPSG:28992", "EPSG:4979").transform(df_cell['X'],
                                                                                               df_cell['Y'])

    # Only keep cell towers in bounding box
    df_cell = df_cell.loc[(df_cell['lon'] >= (model_params["bounding_box"][0] - increase)) & (
                df_cell['lon'] <= model_params["bounding_box"][2] + increase)
                          & (df_cell['lat'] >= model_params["bounding_box"][1] - increase) & (
                                      df_cell['lat'] <= model_params["bounding_box"][3] + increase)]

    # drop rows that contain the partial string "Sci"

    df_cell = df_cell[~df_cell['Hoofdstraalrichting'].str.contains('|'.join(["-"]))]
    df_cell['Hoofdstraalrichting'] = df_cell['Hoofdstraalrichting'].str.replace('\D', '', regex=True)
    df_cell['Hoofdstraalrichting'] = df_cell['Hoofdstraalrichting'].str.replace(' ', '')
    df_cell = df_cell.dropna(subset='POSTCODE')

    # Read in trajectories
    df_trajectory = pd.read_csv(model_params["trajectory_file"])

    # Limit to start and end date
    df_trajectory = df_trajectory.loc[(df_trajectory['timestamp'] >= model_params["start_date"]) & (
                df_trajectory['timestamp'] <= model_params["end_date"])]

    # Create seconds passed column for time sampling
    df_trajectory['seconds'] = [(datetime.strptime(x, "%Y-%m-%d %H:%M:%S") - start).total_seconds() for x in
                                df_trajectory['timestamp']]

    RD = ("+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 "
          "+k=0.999908 +x_0=155000 +y_0=463000 +ellps=bessel "
          "+towgs84=565.237,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812 "
          "+units=m +no_defs")
    WGS84 = '+proj=latlong +datum=WGS84'
    rd_projection = Proj(RD)
    wgs84_projection = Proj(WGS84)
    WGS84_TO_RD = Transformer.from_proj(wgs84_projection, rd_projection)

    df_trajectory['rd.x'],df_trajectory['rd.y'] = WGS84_TO_RD.transform(df_trajectory['cellinfo.wgs84.lon'],df_trajectory['cellinfo.wgs84.lat']) # In the RDPoint and Point definition lon lat is used. Normally lat lon is used. For now I just kept this as it works.
    # Round the locations, because we will not compute the probabilities for every location, only for rounded.
    df_trajectory['rd.x'] = (df_trajectory['rd.x'] / 100).round().astype(int)
    df_trajectory['rd.y'] = (df_trajectory['rd.y'] / 100).round().astype(int)

    # Combine the rounded longitude and latitude into a single list for unique location identification
    rounded_locations = list(zip(df_trajectory['rd.x'],df_trajectory['rd.y']))

    # Store in list for ease of implementation
    all_cells = np.array(list(zip(df_cell['lat'], df_cell['lon'])))

    # Get unique agents
    agents = sorted(pd.unique(df_trajectory['owner']))

    # Load in coverage model, we utilize the model with mnc 8 and 0 time difference
    coverage_models = pickle.load(open(model_params["coverage_file"], 'rb'))

    model = coverage_models[('16', (0, 0))]

    # Initialize variables
    writing_id = 0
    all_grids = []
    all_degree = []
    all_cell_id = []
    distances_home = []
    all_cell_postal_code = []

    # Read in grid with probabilites for each cell in our cell tower dataframe
    for i in range(len(all_cells)):
        grid = model.probabilities(Measurement(
            coords=Point(lat=float(all_cells[i][0]),
                         lon=float(all_cells[i][1])),
            timestamp=datetime.now(),
            postal_code=df_cell['POSTCODE'].iloc[i],
            extra={'mnc': '16',
                   'azimuth': df_cell['Hoofdstraalrichting'].iloc[i],
                   'antenna_id': df_cell['ID'].iloc[i],
                   'postal_code': df_cell['POSTCODE'].iloc[i],
                   'city': df_cell['WOONPLAATSNAAM'].iloc[i]}))
        # Store grids and azimuth degree for later use
        all_grids.append(grid)
        all_degree.append(df_cell['Hoofdstraalrichting'].iloc[i])
        all_cell_id.append(df_cell['ID'].iloc[i])
        all_cell_postal_code.append(df_cell['POSTCODE'].iloc[i])


    # Get unique locations and their counts
    unique_locations = np.unique(rounded_locations, axis = 0)

    # Compute the probabilities of connecting to a cell tower for every rounded location in the dataset.
    probabilities_list = []

    for location in tqdm(unique_locations):
        rd = RDPoint(x=location[0]*100, y=location[1]*100)
        probabilities = [grid.get_value_for_coord(rd) ** 2 for grid in all_grids]
        probabilities_list.append(probabilities)

    df_probabilities = pd.DataFrame({'rd.x':unique_locations[:,0],
                                     'rd.y':unique_locations[:,1],
                                     'probabilities':probabilities_list})
    df_probabilities.set_index(['rd.x','rd.y'],inplace=True)

    # loop over agents and obtain trajectory per agent, store max observed time
    for i in tqdm(range(len(agents))):
        agents_df = df_trajectory[df_trajectory['owner'].isin([agents[i]])]
        max = agents_df['seconds'].iloc[-1]
        agent = re.sub("[^0-9]", "", agents[i])

        # if we do independent sampling then we want to do full sampling twice for each phone
        # else we do the sampling once and utilize switch
        samples = 1
        if (model_params["sampling_method"] == 1):
            samples = 2

        # if we do location based sampling, we keep track of
        if (model_params["sampling_method"] == 3):
            X = np.array(list(zip(agents_df['cellinfo.wgs84.lat'], agents_df['cellinfo.wgs84.lon'])))
            home_loc = agents_df.loc[agents_df['status'] == 'home']
            home = (home_loc['cellinfo.wgs84.lat'].iloc[0], home_loc['cellinfo.wgs84.lon'].iloc[0])
            distances_home = np.linalg.norm(X - np.array((home[0], home[1])), axis=1)

        # for each phone we sample from a poisson distribution with rate event_rate per hour
        for phone in range(samples):
            x_old, y_old = 0, 0
            p_time = 1
            counter = 0

            time_until_event = np.random.default_rng().geometric(p=1 / (3600 * model_params['event_rate']), size=int(
                samples * max * model_params['event_rate'] / (100))).tolist()
            switch_towers = np.random.default_rng().uniform(0, 1, size=int(
                samples * max * model_params['event_rate'] / (100))).tolist()

            time_until_event = np.array(time_until_event).cumsum()
            time_until_event = time_until_event[:np.searchsorted(time_until_event, max, side='right')]
            time_until_event = time_until_event.tolist()

            indices = np.searchsorted(agents_df['seconds'].values, time_until_event, side='right') - 1

            for index in indices:
                x = agents_df['rd.x'].iloc[index]
                y = agents_df['rd.y'].iloc[index]

                if ((x_old != x or y_old != y)
                        or (switch_towers.pop() < model_params["probability_switch"])):
                    probabilities = df_probabilities.at[(x,y),'probabilities']
                    index_cell = choices(list(range(len(probabilities))), weights=probabilities)[0]
                    x_old = x
                    y_old = y

                if (model_params["sampling_method"] == 2):
                    day_time = p_time % 86400
                    if (day_time >= 32400 and day_time <= 61200):
                        phone = 1
                    else:
                        phone = 0

                elif (model_params["sampling_method"] == 3):
                    if (distances_home[index - 1] > 0.05):
                        phone = 1
                    else:
                        phone = 0

                output_writer.writerow([writing_id, f"Agent{agent}", f"{agent}_{phone + 1}",
                                        start + timedelta(seconds=time_until_event[counter]), all_cells[index_cell][0],
                                        all_cells[index_cell][1], all_degree[index_cell], all_cell_id[index_cell],
                                        all_cell_postal_code[index_cell]])

                writing_id += 1
                counter += 1
    output_file.close()


if __name__ == '__main__':
    model_params = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "bounding_box": BOUNDING_BOX,
        "cell_file": CELL_FILE,
        "coverage_file": COVERAGE_FILE,
        "trajectory_file": OUTPUT_TRAJECTORY_FILE,
        "output_file": OUTPUT_CELL_FILE,
        # 1 for independent sampling, 2 for dependent on time and 3 for dependent on location
        "sampling_method": 3,
        "event_rate": 1,  # number of events per hour
        "probability_switch": 0.05  # Probability of switching towers if the phone is stationary
    }
    main(model_params)
