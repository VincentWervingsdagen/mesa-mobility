import pyproj
from src.space.road_network import NetherlandsWalkway
import numpy as np
import geopandas as gpd
import pandas as pd
from datetime import datetime
import networkx as nx
import random
from stochastic.processes.continuous import BrownianBridge
import pickle
from telcell.data.models import Measurement, Point, RDPoint
import scipy


class Particle:
    old_location: [float,float]
    new_location: [float,float]
    weight: float
    path: list[[float,float]]

    def __init__(self,start_point):
        self.old_location = start_point
        self.new_location = start_point # Dummy assignment, so that set_new_location works.

    def set_path(self,path):
        self.path = path

    def set_new_location(self,new_location):
        self.old_location = self.new_location # Sets the old new location to be the old location
        self.new_location = new_location # Sets the new location to be the new location
        # print(self.get_old_location())
        # print(self.get_new_location())

    def determine_weight_distance(self,observation):
        # Simple distance metric,
        # it's probably better to use the coverage model of the NFI with determine_weight_coverage_model.
        distance = np.sqrt((observation['x']-self.new_location[0])**2+(observation['y']-self.new_location[1])**2)*1000
        self.weight = np.min([1,1/distance])

    def determine_weight_coverage_model(self,grid,transformer):
        x,y = transformer.transform(self.new_location[0],self.new_location[1],direction='inverse')
        self.weight = grid.get_value_for_coord(Point(lon = x,lat = y).convert_to_rd())**2

    def get_old_location(self):
        return self.old_location

    def get_new_location(self):
        return self.new_location

    def get_path(self):
        return self.path

    def get_weight(self):
        return self.weight + np.finfo(float).eps # Add small weight so that we do not get divisions by 0

class ParticleFilter:
    walkway: NetherlandsWalkway
    data_crs: str
    graph_crs: str
    particle_list: list[Particle]
    walking_allowed: bool
    bounding_box: list
    observations: pd.DataFrame
    number_of_particles: int
    dict_grids: dict
    transformer: pyproj.Transformer
    list_estimated_location: list[(float,float)]
    list_paths: list[(float,float)]
    return_all_paths: bool

    def __init__(self,
                 roadnetwork_file,
                 coverage_file_path,
                 observation_file,
                 output_file,
                 walking_allowed,
                 bounding_box,
                 N,
                 data_crs="4326",
                 graph_crs="3857",
                 method = 1,
                 return_all_paths = False):
        # Setting
        self.data_crs = data_crs
        self.graph_crs = graph_crs
        self.transformer = pyproj.Transformer.from_crs(crs_from=self.data_crs,crs_to=self.graph_crs,always_xy=True)
        self.walking_allowed = walking_allowed
        self.bounding_box = bounding_box
        self._load_road_vertices_from_file(roadnetwork_file)
        self.load_observations_from_file(observation_file)
        self.number_of_particles = N
        self.coverage_model(coverage_file_path)
        self.list_estimated_location = [(self.observations['cellinfo.wgs84.lon'].iloc[0],self.observations['cellinfo.wgs84.lat'].iloc[0])] # Initialise location on the first celltower.
        self.list_paths = [(self.observations['cellinfo.wgs84.lon'].iloc[0],self.observations['cellinfo.wgs84.lat'].iloc[0])] # Initiliase location on the first cell tower.
        self.return_all_paths = return_all_paths

        #Particle filter
        self.initialise_particles()
        for i in range(len(self.observations)-1):
        #for i in range(5):
            print(i)
            if method == 1:
                self.brownian_bridge(i)
            elif method == 2:
                self.move_particles(i)
            else:
                self.move_particles_version2(i)
            self.compute_weight(i)
            self.estimate_location()
            if method != 1:
                self.resample_particles()

        self.write_to_csv(output_file)



    def _load_road_vertices_from_file(
        self, walkway_file: str
    ) -> None:
        walkway_df = (
                gpd.read_file(walkway_file, self.bounding_box)
                .set_crs(self.data_crs, allow_override=True)
                .to_crs(self.graph_crs)
        )
        if (self.walking_allowed == True):
            walkway_df.loc[walkway_df['maxspeed']==0,'maxspeed'] = 5 # Set walking speed to 5 km/h. This will cause additional roads to be used,
                                                      # so simulation will be slower.
        self.walkway = NetherlandsWalkway(lines=walkway_df[walkway_df['maxspeed']>0]["geometry"],maxspeed=walkway_df[walkway_df['maxspeed']>0]["maxspeed"])

    def load_observations_from_file(self,dataframe_file: str)-> None:
        self.observations = pd.read_csv(dataframe_file,)
        self.observations = self.observations.set_index('cellinfo.id',drop=False)
        self.observations['timestamp'] = pd.to_datetime(self.observations['timestamp'],format='%Y-%m-%d %H:%M:%S')
        self.observations['x'],self.observations['y'] = self.transformer.transform(self.observations['cellinfo.wgs84.lon'],self.observations['cellinfo.wgs84.lat'])

    def coverage_model(self,coverage_model_path):
        coverage_model = open(coverage_model_path, 'rb')
        coverage_models = pickle.load(coverage_model)
        model = coverage_models[('16', (0, 0))]
        df_cell = self.observations[['cellinfo.wgs84.lon','cellinfo.wgs84.lat','cellinfo.azimuth_degrees']].drop_duplicates()
        # Read in grid with probabilites for each cell in our cell tower dataframe
        self.dict_grids = {
            id: model.probabilities(Measurement(
                coords=Point(lon=float(df_cell.at[id, 'cellinfo.wgs84.lon']),
                             lat=float(df_cell.at[id, 'cellinfo.wgs84.lat'])),
                timestamp=datetime.now(),
                extra={'mnc': '16',
                       'azimuth': df_cell.at[id, 'cellinfo.azimuth_degrees'],
                       'antenna_id': id})
            ) for id in df_cell.index
        }

    def initialise_particles(self):
        self.particle_list = [Particle(start_point=(self.walkway.get_nearest_node((self.observations['x'].iloc[0],
                          self.observations['y'].iloc[0])))) for _ in range(self.number_of_particles)]

    def brownian_bridge(self,index): # Generates a brownian bridge between the two cell towers that is not constrained by
        if index < len(self.observations)-1:
            time_difference = (self.observations['timestamp'].iloc[index+1]-self.observations['timestamp'].iloc[index]).total_seconds()
            for i in range(self.number_of_particles):
                new_location = [self.observations['x'].iloc[index], self.observations['y'].iloc[
                    index]]  # Should sample some locations for the new observation from the generative telcell model.
                self.particle_list[i].set_new_location(new_location)
                path_x = BrownianBridge(b=self.particle_list[i].get_new_location()[0]-self.particle_list[i].get_old_location()[0],t=time_difference*1000)
                path_y = BrownianBridge(b=self.particle_list[i].get_new_location()[1]-self.particle_list[i].get_old_location()[1],t=time_difference*1000)
                path = [(x, y) for x, y in zip(path_x.sample(10)+self.particle_list[i].get_old_location()[0], path_y.sample(10)+self.particle_list[i].get_old_location()[1])]
                self.particle_list[i].set_path(path)

    def move_particles(self,index):
        time_difference = (self.observations['timestamp'].iloc[index+1]-self.observations['timestamp'].iloc[index]).total_seconds()
        for i in range(self.number_of_particles): # Horribly inefficient
            old_location = self.particle_list[i].get_new_location()
            dictionary_paths = nx.single_source_dijkstra_path(self.walkway.nx_graph, source=old_location,
                                                                cutoff=time_difference, weight='traversal_time')
            dictionary_key = random.sample(dictionary_paths.keys(), k=1)[0]
            self.particle_list[i].set_path(dictionary_paths.get(dictionary_key))
            self.particle_list[i].set_new_location(dictionary_key)

    def move_particles_version2(self, index): # Voeg een check toe om te kijken of het punt bereikbaar is binnen de tijd.
        location_difference_towers_x = self.observations['x'].iloc[index+1]-self.observations['x'].iloc[index]
        location_difference_towers_y = self.observations['y'].iloc[index+1]-self.observations['y'].iloc[index]
        location_permutation = np.random.uniform(-1,1,size=2*self.number_of_particles)*200 # Maybe we need a smarter guess than 200
        # print(self.observations['x'].iloc[index],self.observations['y'].iloc[index])
        # print(self.observations['x'].iloc[index+1],self.observations['y'].iloc[index+1])
        # print(location_difference_towers_x)
        # print(location_difference_towers_y)
        # print(location_permutation)
        for i in range(self.number_of_particles):
            old_location = self.particle_list[i].get_new_location() # Actually the old location, just confusing name ;(
            new_location = (old_location[0]+location_difference_towers_x+location_permutation[2*i],
                            old_location[1]+location_difference_towers_y+location_permutation[2*i+1])
            self.particle_list[i].set_new_location(self.walkway.get_nearest_node(new_location))
            self.particle_list[i].set_path(nx.shortest_path(G=self.walkway.nx_graph,source=old_location,target=self.particle_list[i].get_new_location(),weight='traversal_time'))
        # print(self.particle_list[0].get_old_location())
        # print(self.particle_list[0].get_new_location())

    def compute_weight(self,index):
        for i in range(self.number_of_particles):
            self.particle_list[i].determine_weight_coverage_model(self.dict_grids.get(self.observations['cellinfo.id'].iloc[index+1]),self.transformer)

    def estimate_location(self):
        weights = np.asarray(list(map(lambda particle: particle.get_weight(), self.particle_list)))
        locations = np.asarray(list(map(lambda particle: particle.get_new_location(), self.particle_list)))
        estimated_coordinates = [np.sum(np.multiply(locations[:,0],weights))/np.sum(weights),np.sum(np.multiply(locations[:,1],weights))/np.sum(weights)]
        self.list_estimated_location.append(self.transformer.transform(*estimated_coordinates,direction='INVERSE'))
        path = [self.transformer.transform(xx=location[0], yy=location[1], direction='INVERSE') for location in self.particle_list[0].get_path()]
        self.list_paths.append(path)

    def resample_particles(self):
        weights = np.asarray(list(map(lambda particle: particle.get_weight(), self.particle_list)))
        print(weights[0:3])
        print(np.sum(weights))
        cumulative_sum = np.cumsum(weights/np.sum(weights))
        cumulative_sum[-1] = 1.  # avoid round-off error
        print(cumulative_sum)
        indexes = np.searchsorted(cumulative_sum, scipy.stats.uniform(0,1).rvs(self.number_of_particles))
        print(indexes)
        self.particle_list = [self.particle_list[index] for index in indexes]

    def write_to_csv(self,output_file):
        self.observations[['estimate_x', 'estimate_y']] = self.list_estimated_location
        self.observations['paths'] = self.list_paths
        self.observations.to_csv(output_file)



