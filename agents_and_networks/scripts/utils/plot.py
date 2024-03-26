import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import re
import os

# Start and End date for trajectory analysis
start_date = '2023-05-01'
end_date = '2023-05-02'

script_dir = os.path.dirname(os.path.abspath(__file__))
df_cell = pd.read_csv(os.path.join(script_dir,'..','..', 'outputs', 'trajectories', 'output_cell.csv'))
df_trajectory = pd.read_csv(os.path.join(script_dir,'..','..', 'outputs', 'trajectories', 'output_trajectory.csv'))
df_trajectory.columns = ['id', 'owner', 'timestamp', 'cellinfo.wgs84.lon', 'cellinfo.wgs84.lat', 'status']
mask = (df_trajectory['timestamp'] >= start_date) & (df_trajectory['timestamp'] <= end_date)
df_trajectory = df_trajectory.loc[mask]

mask = (df_cell['timestamp'] >= start_date) & (df_cell['timestamp'] <= end_date)
df_cell = df_cell.loc[mask]

bounding_box1 = (4.3338,51.9853,4.3658,52.0204)
# bounding_box1 = (4.2490,52.0035,4.4138,52.0934)
# bounding_box2 = (4.8582,52.3478,4.9423,52.3926)


# bounding_box = (min(bounding_box1[0],bounding_box2[0]),min(bounding_box1[1],bounding_box2[1]),
#                              max(bounding_box1[2],bounding_box2[2]),max(bounding_box1[3],bounding_box2[3]))    

walkway_file = os.path.join(script_dir,'..','..', 'data', 'zuid-holland', 'gis_osm_roads_free_1.zip')
walkway_file_trip = os.path.join(script_dir,'..','..', 'data', 'zuid-holland', 'gis_osm_roads_free_1.zip')

# files = [walkway_file,walkway_file_trip]
# boxes = [bounding_box1,bounding_box2]
# walkway_df = gpd.GeoDataFrame(pd.concat([gpd.read_file(i,j) for (i,j) in zip(files,boxes)], 
#                         ignore_index=True))



# motorway_df = gpd.GeoDataFrame(pd.concat([gpd.read_file(i,bounding_box) for i in files], 
#                         ignore_index=True))
# # motorway_df = motorway_df[motorway_df['maxspeed'].isin([100,80,60])]
# motorway_df = motorway_df[motorway_df['fclass'].isin(['motorway','motorway_link','secondary'])]

# walkway_df = gpd.GeoDataFrame(pd.concat([walkway_df,motorway_df]))

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
walkway_df = (
            gpd.read_file(walkway_file, bounding_box1)
        )

# files = [walkway_file,walkway_file_trip]

# walkway_df = gpd.GeoDataFrame(pd.concat([gpd.read_file(i,bounding_box) for i in files], 
#                         ignore_index=True))
walkway_df.plot(ax=ax1,color='black',linewidth=0.5)
walkway_df.plot(ax=ax2,color='black',linewidth=0.5)
walkway_df.plot(ax=ax3,color='black',linewidth=0.5)

agents = sorted(pd.unique(df_cell['owner']))
print(agents)

for i in range(0,3):
    agents_cell = df_cell[df_cell['owner'].isin([agents[i]])] 
    agents_trajectory = df_trajectory[df_trajectory['owner'].isin([agents[i]])] 

    lon = agents_trajectory['cellinfo.wgs84.lon']
    lat = agents_trajectory['cellinfo.wgs84.lat']
    ax1.plot(lon, lat, zorder=5,linewidth=1.5)
    ax1.scatter(lon, lat, zorder=10, s=10)
    ax1.set_title('trajectories')

    phone1_df = agents_cell[agents_cell['device'].isin([re.sub("[^0-9]", "", agents[i])+"_1"])] 
    phone2_df = agents_cell[agents_cell['device'].isin([re.sub("[^0-9]", "", agents[i])+"_2"])] 


    lon1 = phone1_df['cellinfo.wgs84.lon']
    lat1 = phone1_df['cellinfo.wgs84.lat']
    lon2 = phone2_df['cellinfo.wgs84.lon']
    lat2 = phone2_df['cellinfo.wgs84.lat']

    ax2.plot(lon1, lat1, zorder=5,linewidth=1.2)
    ax2.scatter(lon1, lat1, zorder=10, s=10)
    ax2.set_title('Cell Towers: Phone 1')
    ax3.plot(lon2, lat2, zorder=5,linewidth=1.2)
    ax3.scatter(lon2, lat2, zorder=10, s=10)
    ax3.set_title('Cell Towers: Phone 2')

plt.show()