import pickle
import pandas as pd
import numpy as np
import re
import csv

from pyproj import Transformer
from datetime import datetime, timedelta
from telcell.data.models import Measurement, Point,RDPoint
from random import choices
from config import BOUNDING_BOX, BOUNDING_INCREASE, START_DATE, END_DATE, OUTPUT_TRAJECTORY_FILE, OUTPUT_CELL_FILE, CELL_FILE, COVERAGE_FILE

import os

script_dir = os.path.dirname(os.path.abspath(__file__))

"""
Script to obtain the cell tower samplings from a pre-existing coverage model
"""
def main(model_params):

    # Retrieve start date
    start = datetime.strptime(model_params["start_date"],"%Y-%m-%d")

    # Expand bounding box size
    increase = BOUNDING_INCREASE

    # Setup output file
    output_file = open(model_params["output_file"], 'w')
    output_writer = csv.writer(output_file)
    output_writer.writerow(['id','owner','device','timestamp','cellinfo.wgs84.lat','cellinfo.wgs84.lon','cellinfo.azimuth_degrees','cellinfo.id','cellinfo.postal_code'])

    # Read in cell towers: 
    df_cell = pd.read_csv(model_params["cell_file"])

    # As coverage model does not take certain information into account (such as safe distance and power) 
    # we remove this information and drop any duplicates.
    df_cell = df_cell.drop(['Samenvatting','Vermogen', 'Frequentie','Veilige afstand','id'], axis=1)
    df_cell = df_cell.drop_duplicates()

    # Only consider LTE (4G) for now
    df_cell = df_cell.loc[df_cell['HOOFDSOORT'] == "LTE"]

    # Transform to wgs84
    df_cell['lat'],df_cell['lon'] =  Transformer.from_crs("EPSG:28992","EPSG:4979").transform(df_cell['X'],df_cell['Y'])
 
    # Only keep cell towers in bounding box
    df_cell = df_cell.loc[(df_cell['lon'] >= (model_params["bounding_box"][0]-increase)) & (df_cell['lon'] <= model_params["bounding_box"][2]+increase)
                          & (df_cell['lat'] >= model_params["bounding_box"][1]-increase) & (df_cell['lat'] <= model_params["bounding_box"][3]+increase)]

    # drop rows that contain the partial string "Sci"
    
    df_cell = df_cell[~df_cell['Hoofdstraalrichting'].str.contains('|'.join(["-"]))]
    df_cell['Hoofdstraalrichting'] = df_cell['Hoofdstraalrichting'].str.replace('\D', '',regex=True)
    df_cell['Hoofdstraalrichting'] = df_cell['Hoofdstraalrichting'].str.replace(' ', '')

    # Read in trajectories
    df_trajectory = pd.read_csv(model_params["trajectory_file"])

    # Limit to start and end date
    df_trajectory = df_trajectory.loc[(df_trajectory['timestamp'] >= model_params["start_date"]) & (df_trajectory['timestamp'] <= model_params["end_date"])]
    
    # Create seconds passed column for time sampling
    df_trajectory['seconds'] = [(datetime.strptime(x,"%Y-%m-%d %H:%M:%S") - start).total_seconds() for x in df_trajectory['timestamp']]

    # Store in list for ease of implementation
    all_cells = np.array(list(zip(df_cell['lat'],df_cell['lon'])))

    # Get unique agents
    agents = sorted(pd.unique(df_trajectory['owner']))  
    
    # Load in coverage model, we utilize the model with mnc 8 and 0 time difference
    coverage_models = pickle.load(open(model_params["coverage_file"], 'rb'))

    model = coverage_models[('16',(0, 0))]

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

    # loop over agents and obtain trajectory per agent, store max observed time
    for i in range(len(agents)):
        agents_df = df_trajectory[df_trajectory['owner'].isin([agents[i]])] 
        max = agents_df['seconds'].iloc[-1]

        print("Agents",i)
        
        # if we do independent sampling then we want to do full sampling twice for each phone
        # else we do the sampling once and utilize switch
        samples = 1
        if (model_params["sampling_method"] == 1):
            samples = 2
        
        # if we do location based sampling, we keep track of 
        if (model_params["sampling_method"] == 3):
            X = np.array(list(zip(agents_df['cellinfo.wgs84.lat'],agents_df['cellinfo.wgs84.lon'])))    
            home_loc = agents_df.loc[agents_df['status'] == 'home']
            home = (home_loc['cellinfo.wgs84.lat'].iloc[0],home_loc['cellinfo.wgs84.lon'].iloc[0])
            distances_home = np.linalg.norm(X-np.array((home[0],home[1])), axis=1)

        time_until_event = np.random.default_rng().geometric(p=1/(3600*model_params['event_rate']), size=int(samples*max*model_params['event_rate']/(100))).tolist()
        switch_towers = np.random.default_rng().uniform(0,1, size=int(samples*max*model_params['event_rate']/(100))).tolist()
        print("Agents",i)

        # for each phone we sample from a poisson distribution with rate event_rate per hour
        for phone in range(samples):
            index = 0
            x_old, y_old = 0, 0
            p_time = 1

            while(p_time <= max):
                while (agents_df['seconds'].iloc[index] < p_time):
                    index += 1
                x = agents_df['cellinfo.wgs84.lat'].iloc[index-1]
                y = agents_df['cellinfo.wgs84.lon'].iloc[index-1]
                rd = Point(lat = x,lon = y).convert_to_rd()
                
                if ((x_old != round(rd.x/100)*100 or y_old != round(rd.y/100)*100)
                or (switch_towers.pop() < model_params["probability_switch"])):
                    # Take the square of the probabilities so that closer cell towers will be selected more often.
                    probabilities = [grid.get_value_for_coord(RDPoint(x=rd.x,y = rd.y))**2 for grid in all_grids]
                    index_cell = choices(list(range(len(probabilities))), weights = probabilities)[0]
                    x_old = round(rd.x/100)*100
                    y_old = round(rd.y/100)*100


                agent = re.sub("[^0-9]", "", agents[i])
                if (model_params["sampling_method"] == 2):
                    day_time = p_time%86400
                    if (day_time >= 32400 and day_time <= 61200):
                        phone = 1
                    else:
                        phone = 0
                    
                elif (model_params["sampling_method"] == 3):
                    if(distances_home[index-1] > 0.05):
                        phone = 1
                    else:
                        phone = 0
       
                output_writer.writerow([writing_id, f"Agent{agent}", f"{agent}_{phone+1}", 
                                    start + timedelta(seconds = p_time), all_cells[index_cell][0], all_cells[index_cell][1], all_degree[index_cell],all_cell_id[index_cell],all_cell_postal_code[index_cell]])
                
                p_time += time_until_event.pop(1)
                writing_id += 1

    output_file.close()



if __name__ == '__main__':
    model_params = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "bounding_box":BOUNDING_BOX,
        "cell_file": CELL_FILE,
        "coverage_file": COVERAGE_FILE,
        "trajectory_file": OUTPUT_TRAJECTORY_FILE,
        "output_file": OUTPUT_CELL_FILE,
        # 1 for independent sampling, 2 for dependent on time and 3 for dependent on location
        "sampling_method": 1,
        "event_rate": 1,  # number of events per hour
        "probability_switch": 0.1  # Probability of switching towers if the phone is stationary
    }
    main(model_params)
