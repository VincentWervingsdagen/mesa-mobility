import pandas as pd
import numpy as np
import re
import csv
from pyproj import Transformer
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from math import atan2,degrees

import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def main(model_params):
    # Retrieve start date
    start = datetime.strptime(model_params["start_date"],"%Y-%m-%d")

    # Setup output file
    output_file = open(model_params["output_file"], 'w')
    output_writer = csv.writer(output_file)
    output_writer.writerow(['id','owner','device','timestamp','cellinfo.wgs84.lon','cellinfo.wgs84.lat','cellinfo.azimuth_degrees','frequency','height','power'])

    # Read in cell towers, transform lon/lat, limit to bounding box, format the orientation data
    df_cell = pd.read_csv(model_params["cell_file"])
    df_cell = df_cell[df_cell['sat_code']=='LTE']
    df_cell['Vermogen'] = df_cell['Vermogen'].str.replace(' dBW', '').astype(float)
    df_cell['lon'],df_cell['lat'] =  Transformer.from_crs("EPSG:28992","EPSG:4979").transform(df_cell['X'],df_cell['Y'])
    df_cell = df_cell.loc[(df_cell['lat'] >= model_params["bounding_box"][0]) & (df_cell['lat'] <= model_params["bounding_box"][2]) 
                          & (df_cell['lon'] >= model_params["bounding_box"][1]) & (df_cell['lon'] <= model_params["bounding_box"][3])]
    df_cell['Hoofdstraalrichting'] = df_cell['Hoofdstraalrichting'].str.replace('\D', '',regex=True)
    df_cell['Hoofdstraalrichting'] = df_cell['Hoofdstraalrichting'].str.replace(' ', '')

    print(df_cell.columns)
    # Read in trajectories, limit and add seconds passed column
    df_trajectory = pd.read_csv(model_params["trajectory_file"])
    df_trajectory.columns = ['id','owner','timestamp','cellinfo.wgs84.lon','cellinfo.wgs84.lat','status','speed']
    df_trajectory = df_trajectory.loc[(df_trajectory['timestamp'] >= model_params["start_date"]) & (df_trajectory['timestamp'] <= model_params["end_date"])]
    df_trajectory['seconds'] = [(datetime.strptime(x,"%Y-%m-%d %H:%M:%S") - start).total_seconds() for x in df_trajectory['timestamp']]
    # df_trajectory['seconds_adjusted_speed'] = np.hstack((0,np.cumsum(np.diff(df_trajectory['seconds'])*(df_trajectory['speed'][0:-1]+1))))#Events will happen more often while moving to model base station handover.

    all_cells = np.array(list(zip(df_cell['lat'],df_cell['lon'])))

    agents = sorted(pd.unique(df_trajectory['owner']))  
    writing_id = 0

    # loop over agents and obtain trajectory per agent
    for i in range(len(agents)):
        agents_df = df_trajectory[df_trajectory['owner'].isin([agents[i]])] 
        max = agents_df['seconds'].iloc[-1]

        time_until_event = np.random.default_rng().geometric(p=1/(60*model_params['event_rate']), size=int(max/10)).tolist()
        for phone in range(1):
            index = 0
            p_time = 1
            x_old, y_old = 0,0
            cellx, celly, degree = 0,0,0

            while(p_time <= max):
                while (agents_df['seconds'].iloc[index] < p_time):
                    index += 1
                x = agents_df['cellinfo.wgs84.lon'].iloc[index-1]
                y = agents_df['cellinfo.wgs84.lat'].iloc[index-1]

                if (x_old != x or y_old != y):
                    position = np.array((x,y))
                    distances = (np.linalg.norm(all_cells-position, axis=1))
                    found = False

                    while (not found):
                        index_dis = np.argmin(distances)
                        degree_cell = df_cell['Hoofdstraalrichting'].iloc[index_dis]
                        degree_actual = (degrees(atan2(y-all_cells[index_dis][1], x-all_cells[index_dis][0]))+360)%360
                        if (degree_actual >= int(degree_cell) - 60 and degree_actual <= int(degree_cell) + 60):
                            found = True
                        else:
                            distances[index_dis] = float('inf')

                    cellx = all_cells[index_dis][0]
                    celly = all_cells[index_dis][1]
                    degree = degree_cell
                    x_old = x
                    y_old = y

                    frequency = df_cell['Frequentie'].iloc[index_dis]
                    height = df_cell['Hoogte'].iloc[index_dis]
                    power = df_cell['Vermogen'].iloc[index_dis]

                # if (x_old != x or y_old != y):
                #     actual_degree = (np.arctan2(y-all_cells[:,1],x-all_cells[:,0])*180/np.pi + 360) % 360
                #     connection_possible = ((actual_degree >= df_cell['Hoofdstraalrichting'].astype(int)-60)&(actual_degree <= df_cell['Hoofdstraalrichting'].astype(int)+60))
                #     position = np.array((x, y))
                #     distances = (np.linalg.norm(all_cells[connection_possible] - position, axis=1)) / (
                #                 (df_cell[connection_possible]['Vermogen'].to_numpy() / 10) ** 2)
                #     index_dis = np.argmin(distances)
                #
                #     cellx = all_cells[index_dis][0]
                #     celly = all_cells[index_dis][1]
                #     degree = actual_degree[index_dis]
                #     x_old = x
                #     y_old = y
                #     frequency = df_cell['Frequentie'].iloc[index_dis]
                #     height = df_cell['Hoogte'].iloc[index_dis]
                #     power = df_cell['Vermogen'].iloc[index_dis]

                agent = re.sub("[^0-9]", "", agents[i])

                output_writer.writerow([writing_id, f"Agent{agent}", f"{agent}_{phone+1}",
                    start + timedelta(seconds = p_time), cellx, celly, degree,frequency,height,power])

                p_time += time_until_event.pop(1)//(1+df_trajectory['speed'].iloc[index-1])
                writing_id += 1
    output_file.close()



if __name__ == '__main__':
    model_params = {
        "start_date": '2023-05-01',
        "end_date": '2023-05-02',
        #"bounding_box":(4.3338,51.9853,4.3658,52.0204), #Delft
        "bounding_box": (4.1874, 51.8280, 4.593, 52.0890), #Noord en zuid holland
        "cell_file": os.path.join(script_dir,'..', '..', 'data', '20191202131001.csv'),
        "trajectory_file": os.path.join(script_dir,'..', '..', 'outputs', 'trajectories','output_trajectory_7hours.csv'),
        "output_file": os.path.join(script_dir,'..', '..', 'outputs', 'trajectories','output_cell.csv'),
        "event_rate": 10 #number of events per hour
    }
    main(model_params)