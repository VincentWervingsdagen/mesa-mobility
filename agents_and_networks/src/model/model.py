import uuid
from functools import partial
import geopandas as gpd
import numpy as np
import mesa
import mesa_geo as mg
import pandas as pd
from shapely.geometry import Point
import csv

from datetime import datetime, timedelta

import os

from src.agent.building import Building
from src.agent.commuter import Commuter
from src.space.netherlands import Netherlands
from src.space.road_network import NetherlandsWalkway
from shapely.geometry import Point
from scipy.stats import poisson
import random
from pyproj import Transformer


def get_time(model) -> pd.Timedelta:
    return pd.Timedelta(days=model.day, hours=model.hour, minutes=model.minute)


def get_num_commuters_by_status(model, status: str) -> int:
    commuters = [
        commuter for commuter in model.schedule.agents if commuter.status == status
    ]
    return len(commuters)


def get_average_visited_locations(model) -> list:
    commuters = [
        len(commuter.visited_locations) for commuter in model.schedule.agents 
    ]
    return sum(commuters)/len(commuters)


class AgentsAndNetworks(mesa.Model):
    schedule: mesa.time.RandomActivation
    output_file: csv
    start_date: str
    current_id: int
    space: Netherlands
    walkway: NetherlandsWalkway
    bounding_box:list
    num_commuters: int
    step_duration: int
    alpha: float
    tau_jump: float    # in meters
    tau_jump_min: float
    beta: float
    tau_time: float    # in hours
    tau_time_min: float
    rho: float
    gamma: float
    day: int
    hour: int
    minute: int
    second: int
    positions_to_write: list[int,datetime.timestamp,float,float,str,float]
    positions: list[float,float]
    writing_id_trajectory:int
    common_work: Building
    datacollector: mesa.DataCollector

    def __init__(
        self,
        data_crs: str,
        buildings_file: str,
        walkway_file: str,
        num_commuters,
        step_duration,
        alpha,
        tau_jump,   # in meters
        tau_jump_min,
        beta,
        tau_time,   # in hours
        tau_time_min,
        rho,
        gamma,
        bounding_box,
        walking_allowed: bool,
        model_crs="epsg:3857",
        start_date="2023-05-01",
    ) -> None:
        super().__init__()
        self.schedule = mesa.time.RandomActivation(self)
        self.start_date = datetime.strptime(start_date,"%Y-%m-%d")
        self.data_crs = data_crs
        self.space = Netherlands(crs=model_crs)
        self.num_commuters = num_commuters
        self.space.number_commuters = num_commuters
        self.bounding_box = bounding_box
        self.step_duration = step_duration
        self.walking_allowed = walking_allowed
        self.positions_to_write = []
        self.positions = []

        Commuter.speed = 0. #Updates the current speed so that it can be saved.
        Commuter.ALPHA = alpha
        Commuter.TAU_jump = tau_jump
        Commuter.TAU_jump_min = tau_jump_min
        Commuter.BETA = beta
        Commuter.TAU_time = tau_time
        Commuter.TAU_time_min = tau_time_min
        Commuter.RHO = rho
        Commuter.GAMMA = gamma

        print("read in buildings file")
        self._load_buildings_from_file(buildings_file, crs=model_crs)
        print("read in road file")
        self._load_road_vertices_from_file(walkway_file, crs=model_crs)

        self._set_building_entrance()
        self.day = 0
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.writing_id_trajectory = 0
        self._create_commuters()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file_trajectory = open(os.path.join(script_dir, '..','..', 'outputs', 'trajectories', 'output_trajectory.csv'), 'w')
        csv.writer(output_file_trajectory).writerow(['id','owner','timestamp','cellinfo.wgs84.lon','cellinfo.wgs84.lat','status','speed'])

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "time": get_time,
                "status_home": partial(get_num_commuters_by_status, status="home"),
                "status_work": partial(get_num_commuters_by_status, status="work"),
                "status_traveling": partial(
                    get_num_commuters_by_status, status="transport"
                ),
                "average_visited_locations": partial(
                    get_average_visited_locations
                ),
            }
        )
        self.datacollector.collect(self)
        
        
    def _create_commuters(self) -> None:
        date = self.start_date
        for i in range(self.num_commuters):
            random_home = self.space.get_random_building()
            commuter = Commuter(
                unique_id=uuid.uuid4().int,
                model=self,
                geometry=Point(random_home.centroid),
                crs=self.space.crs,
            )
            commuter.set_home(random_home)
            commuter.set_next_location(commuter.my_home)
            random_home.visited = True
            commuter.set_visited_location(random_home,25)
            commuter.status = "home"
            commuter.speed = 0.
            self.space.add_commuter(commuter, True)
            self.schedule.add(commuter)
            self.positions.append([commuter.geometry.x,commuter.geometry.y])
            self.positions_to_write.append([i,date,commuter.geometry.x,commuter.geometry.y,commuter.status,commuter.speed])

    def _load_buildings_from_file(
        self, buildings_file: str, crs: str
    ) -> None:
        # read in buildings from normal bounding box. If it is a large file>500000 buildings~half of zuid holland
        # then there will be 2000 to 10000 buildings, for smaller bounding boxes it will be less items.
        # Performance improvement compared to resampling,
        # although resampling is better if you want random buildings instead of deterministic buildings.
        random_integer = random.randint(100,500)
        buildings_df = gpd.read_file(buildings_file, bbox=(self.bounding_box),rows=slice(0,100000*random_integer+1,random_integer))
        print("number buildings: ",len(buildings_df))

        buildings_df.index.name = "unique_id"
        buildings_df = buildings_df.set_crs(self.data_crs, allow_override=True).to_crs(
            crs
        )
        buildings_df["centroid"] = [
            (x, y) for x, y in zip(buildings_df.centroid.x, buildings_df.centroid.y)
        ]
        building_creator = mg.AgentCreator(Building, model=self)
        buildings = building_creator.from_GeoDataFrame(buildings_df)
        self.space.add_buildings(buildings)

    def _load_road_vertices_from_file(
        self, walkway_file: str, crs: str
    ) -> None:
        walkway_df = (
                gpd.read_file(walkway_file, self.bounding_box)
                .set_crs(self.data_crs, allow_override=True)
                .to_crs(crs)
        )
        if (self.walking_allowed == True):
            walkway_df.loc[walkway_df['maxspeed']==0,'maxspeed'] = 5 # Set walking speed to 5 km/h. This will cause additional roads to be used,
                                                      # so simulation will be slower.
        self.walkway = NetherlandsWalkway(lines=walkway_df[walkway_df['maxspeed']>0]["geometry"],maxspeed=walkway_df[walkway_df['maxspeed']>0]["maxspeed"])


    def _set_building_entrance(self) -> None:
        for building in (
            *self.space.buildings,
        ):
            building.entrance_pos = self.walkway.get_nearest_node(building.centroid)



    def step(self) -> None:
        self.__update_clock()
        self.schedule.step()

        total_seconds = self.day*24*60*60 + self.hour*60*60 + self.minute*60 + self.second
        time = self.start_date + timedelta(seconds = total_seconds)

        for i in range(self.num_commuters):
            commuter = self.schedule.agents[i]
            x = commuter.geometry.x
            y = commuter.geometry.y
            if (self.positions[i][0] != x or self.positions[i][1] != y):
                self.positions_to_write.append([i,time,x,y,commuter.status,commuter.speed])
                self.positions[i][0] = x
                self.positions[i][1] = y


        if (total_seconds/self.step_duration == 60):
            self.__write_to_file()
            self.positions_to_write = []
                
    
    def __write_to_file(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = open(os.path.join(script_dir, '..','..', 'outputs', 'trajectories', 'output_trajectory.csv'), 'a')
        output_writer = csv.writer(output_file)
        for pos in self.positions_to_write:
            lon,lat = Transformer.from_crs("EPSG:3857","EPSG:4326").transform(pos[2],pos[3])
            output_writer.writerow([self.writing_id_trajectory, f"Agent{pos[0]}", 
                                   pos[1],
                                   lat,lon,pos[4],pos[5]])
            self.writing_id_trajectory += 1
        output_file.close()
        total_seconds = self.day*24*60*60 + self.hour*60*60 + self.minute*60 + self.second
        time = self.start_date + timedelta(seconds = total_seconds)
        print("time: ",time)
        print("average locations: ",get_average_visited_locations(self))
        


    def __update_clock(self) -> None:
        self.second += self.step_duration
        if self.second >= 60:
            while self.second/60 >= 1:
                self.minute += 1
                self.second -= 60
            if self.minute >= 60:
                while self.minute/60 >= 1:
                    self.hour += 1
                    self.minute -= 60
                if self.hour >= 24:
                    self.day += 1
                    self.hour = 0
        
                
        
               
                    