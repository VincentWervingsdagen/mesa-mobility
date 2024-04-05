import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import re
import os
from matplotlib.widgets import Slider, Button


# Start and End date for trajectory analysis
start_date = '2023-05-01'
end_date = '2023-05-02'

script_dir = os.path.dirname(os.path.abspath(__file__))
df_cell = pd.read_csv(os.path.join(script_dir,'..','..', 'outputs', 'trajectories', 'output_cell.csv'))
df_trajectory = pd.read_csv(os.path.join(script_dir,'..','..', 'outputs', 'trajectories', 'output_trajectory_12hours.csv'))

df_trajectory.columns = ['id', 'owner', 'timestamp', 'cellinfo.wgs84.lon', 'cellinfo.wgs84.lat', 'status']
mask = (df_trajectory['timestamp'] >= start_date) & (df_trajectory['timestamp'] <= end_date)
df_trajectory = df_trajectory.loc[mask]

mask = (df_cell['timestamp'] >= start_date) & (df_cell['timestamp'] <= end_date)
df_cell = df_cell.loc[mask]

#"bounding_box":(4.3338,51.9853,4.3658,52.0204), #Delft
bounding_box = (4.1874, 51.8280, 4.593, 52.0890) #Noord en zuid holland

walkway_file = os.path.join(script_dir,'..','..', 'data', 'zuid-holland', 'gis_osm_roads_free_1.zip')
walkway_file_trip = os.path.join(script_dir,'..','..', 'data', 'noord-holland', 'gis_osm_roads_free_1.zip')

# files = [walkway_file,walkway_file_trip]
# boxes = [bounding_box1,bounding_box2]
# walkway_df = gpd.GeoDataFrame(pd.concat([gpd.read_file(i,j) for (i,j) in zip(files,boxes)], 
#                         ignore_index=True))



# motorway_df = gpd.GeoDataFrame(pd.concat([gpd.read_file(i,bounding_box) for i in files], 
#                         ignore_index=True))
# # motorway_df = motorway_df[motorway_df['maxspeed'].isin([100,80,60])]
# motorway_df = motorway_df[motorway_df['fclass'].isin(['motorway','motorway_link','secondary'])]

# walkway_df = gpd.GeoDataFrame(pd.concat([walkway_df,motorway_df]))

agents = sorted(pd.unique(df_cell['owner']))
print(agents)

mask = pd.to_datetime(df_cell['timestamp']).dt.round('min')
df_trajectory['timestamp2'] = pd.to_datetime(df_trajectory['timestamp'])
df_trajectory = df_trajectory.set_index('timestamp2')
df_trajectory = df_trajectory.resample('min').ffill()

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.set_xlim(min(df_trajectory['cellinfo.wgs84.lon'])-0.01, max(df_trajectory['cellinfo.wgs84.lon'])+0.01)
ax1.set_ylim(min(df_trajectory['cellinfo.wgs84.lat'])-0.01, max(df_trajectory['cellinfo.wgs84.lat'])+0.01)

ax2.set_xlim(min(df_cell['cellinfo.wgs84.lon'])-0.01, max(df_cell['cellinfo.wgs84.lon'])+0.01)
ax2.set_ylim(min(df_cell['cellinfo.wgs84.lat'])-0.01, max(df_cell['cellinfo.wgs84.lat'])+0.01)

ax3.set_xlim(min(df_trajectory['cellinfo.wgs84.lon'])-0.01, max(df_trajectory['cellinfo.wgs84.lon'])+0.01)
ax3.set_ylim(min(df_trajectory['cellinfo.wgs84.lat'])-0.01, max(df_trajectory['cellinfo.wgs84.lat'])+0.01)

walkway_df = (gpd.read_file(walkway_file, bounding_box,engine="pyogrio"))

# files = [walkway_file,walkway_file_trip]

# walkway_df = gpd.GeoDataFrame(pd.concat([gpd.read_file(i,bounding_box) for i in files],
#                         ignore_index=True))
walkway_df.plot(ax=ax1,color='black',linewidth=0.5)
walkway_df.plot(ax=ax2,color='black',linewidth=0.5)
walkway_df.plot(ax=ax3,color='black',linewidth=0.5)

for i in range(0,1):
    agents_cell = df_cell[df_cell['owner'].isin([agents[i]])]
    agents_trajectory = df_trajectory[df_trajectory['owner'].isin([agents[i]])]

    lon = agents_trajectory['cellinfo.wgs84.lon']
    lat = agents_trajectory['cellinfo.wgs84.lat']

    lon_est = agents_cell['estimate_lon']
    lat_est = agents_cell['estimate_lat']

    phone1_df = agents_cell[agents_cell['device'].isin([re.sub("[^0-9]", "", agents[i])+"_1"])]
    phone2_df = agents_cell[agents_cell['device'].isin([re.sub("[^0-9]", "", agents[i])+"_2"])]

    lon1 = phone1_df['cellinfo.wgs84.lon']
    lat1 = phone1_df['cellinfo.wgs84.lat']

    ax1.plot(lon, lat, zorder=5,linewidth=2,color='blue')
    ax1.scatter(lon, lat, zorder=10, s=10,color='blue')
    ax1.set_title('trajectories')

    ax2.scatter(lon1, lat1, zorder=10,color='orange',s=50)
    ax2.set_title('Cell Towers: Phone')

    ax3.plot(lon_est, lat_est, zorder=5,linewidth=1.5, color='red')
    ax3.scatter(lon_est, lat_est, zorder=10,s=10,color = 'red')
    ax3.plot(lon, lat, zorder=5,linewidth=1.5,color='blue')
    ax3.scatter(lon, lat, zorder=10,s=10,color='blue')
    ax3.set_title('estimate path')

    plt.show()

    lon = agents_trajectory.loc[mask]['cellinfo.wgs84.lon']
    lat = agents_trajectory.loc[mask]['cellinfo.wgs84.lat']

    lon1 = phone1_df['cellinfo.wgs84.lon']
    lat1 = phone1_df['cellinfo.wgs84.lat']

    fig, ax = plt.subplots()
    ax.set_xlim(min(df_trajectory['cellinfo.wgs84.lon'])-0.01, max(df_trajectory['cellinfo.wgs84.lon'])+0.01)
    ax.set_ylim(min(df_trajectory['cellinfo.wgs84.lat'])-0.01, max(df_trajectory['cellinfo.wgs84.lat'])+0.01)

    walkway_df.plot(ax=ax, color='black', linewidth=0.5,alpha=0.5)
    # Scatter plot of initial data
    scatter = ax.scatter(lon, lat, label='Phone Location', color='blue',alpha=1)
    scatter2 = ax.scatter(lon1, lat1, label='Tower Location', color='orange',alpha=1)
    scatter3 = ax.scatter(lon_est,lat_est, label='Estimated Location', color='red',alpha=1)

    ax.legend()

    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.15, 0.02, 0.65, 0.03], facecolor=axcolor)
    stime = Slider(axtime, 'Time', 0, len(lon) - 1, valinit=0, valstep=1)

    # Define button callbacks
    def button_forward(event):
        stime.set_val(stime.val + 1)


    def button_backward(event):
        stime.set_val(stime.val - 1)

    # Add buttons
    axprev = plt.axes([0.02, 0.02, 0.1, 0.03])
    axnext = plt.axes([0.88, 0.02, 0.1, 0.03])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(button_forward)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(button_backward)


    def update(val):
        time_index = int(stime.val)
        scatter.set_offsets([[lon[time_index], lat[time_index]]])
        scatter2.set_offsets([[lon1[time_index], lat1[time_index]]])
        scatter3.set_offsets([[lon_est[time_index], lat_est[time_index]]])
        fig.canvas.draw_idle()


    stime.on_changed(update)

    plt.show()


