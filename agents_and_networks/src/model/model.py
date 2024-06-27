import uuid
from functools import partial
import geopandas as gpd

import mesa
import mesa_geo as mg
import pandas as pd

import csv
import os

from datetime import datetime, timedelta

from src.agent.building import Building
from src.agent.commuter import Commuter
from src.space.netherlands import Netherlands
from src.space.road_network import NetherlandsWalkway
from shapely.geometry import Point

import random
from pyproj import Transformer


def ask_overwrite_permission(file_path):
    if os.path.exists(file_path):
        response = input(f"The file '{file_path}' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
        if response != 'y':
            print("Operation aborted by the user.")
            return False
    return True


def get_time(model) -> pd.Timedelta:
    return pd.Timedelta(days=model.time.day, hours=model.time.hour, minutes=model.time.minute)


def get_num_commuters_by_status(model, status: str) -> int:
    commuters = [
        commuter for commuter in model.schedule.agents if commuter.status == status
    ]
    return len(commuters)


def get_average_visited_locations(model) -> float:
    commuters = [
        len(commuter.visited_locations) for commuter in model.schedule.agents 
    ]
    return sum(commuters)/len(commuters)


class AgentsAndNetworks(mesa.Model):
    schedule: mesa.time.RandomActivation
    output_file_trajectory: csv
    output_file: str
    end_date: datetime
    time: datetime
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
    step_counter: int
    positions_to_write: list[int,datetime.timestamp,float,float,str,float]
    positions: list[float,float]
    writing_id_trajectory:int
    common_work_building: Building
    datacollector: mesa.DataCollector
    walking_allowed:bool
    common_work:bool


    def __init__(
        self,
        data_crs: str,
        buildings_file: str,
        walkway_file: str,
        output_file: str,
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
        common_work: bool,
        model_crs="epsg:3857",
        start_date="2023-05-01",
        end_date="2023-06-01"
    ) -> None:
        super().__init__()
        self.schedule = mesa.time.RandomActivation(self)
        self.time = datetime.strptime(start_date,"%Y-%m-%d")
        self.end_date = datetime.strptime(end_date,"%Y-%m-%d")
        self.data_crs = data_crs
        self.space = Netherlands(crs=model_crs)
        self.num_commuters = num_commuters
        self.space.number_commuters = num_commuters
        self.bounding_box = bounding_box
        self.step_duration = step_duration
        self.walking_allowed = walking_allowed
        self.common_work=common_work
        self.positions_to_write = []
        self.positions = []
        self.output_file = output_file

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
        self.writing_id_trajectory = 0
        self.step_counter = 0
        self._create_commuters()

        if ask_overwrite_permission(output_file):
            self.output_file_trajectory = open(self.output_file, 'w')
            csv.writer(self.output_file_trajectory).writerow(['id','owner','timestamp','cellinfo.wgs84.lon','cellinfo.wgs84.lat','status','speed'])
            self.output_file_trajectory.close()
        else:
            print('File cannot be overwritten')
            exit()

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
        if self.common_work:
            self.common_work_building = self.space.get_random_building()
            self.common_work_building.visited = True
        else:
            pass
        for i in range(self.num_commuters):
            random_home = self.space.get_random_building()
            commuter = Commuter(
                unique_id=uuid.uuid4().int,
                model=self,
                geometry=Point(random_home.centroid),
                crs=self.space.crs,
            )
            commuter.set_home(random_home)
            random_home.visited = True
            if self.common_work:
                commuter.set_work(self.common_work_building)
            else:
                pass
            commuter.status = "home"
            commuter.speed = 0.
            self.space.add_commuter(commuter, True)
            self.schedule.add(commuter)
            self.positions.append([commuter.geometry.x,commuter.geometry.y])
            self.positions_to_write.append([i,self.time,commuter.geometry.x,commuter.geometry.y,commuter.status,commuter.speed])

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
            walkway_df.loc[walkway_df['maxspeed']==0,'maxspeed'] = 5 # Set walking speed to 5 km/h. This will cause additional roads and paths to be used,
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
        self.step_counter += 1
        time = self.time
        for i in range(self.num_commuters):
            commuter = self.schedule.agents[i]
            x = commuter.geometry.x
            y = commuter.geometry.y
            if (self.positions[i][0] != x or self.positions[i][1] != y):
                self.positions_to_write.append([i,time,x,y,commuter.status,commuter.speed])
                self.positions[i][0] = x
                self.positions[i][1] = y
            if self.step_counter == 60:
                self.positions_to_write.append([i,time, x, y, commuter.status, commuter.speed])

        if self.step_counter == 60:
            self.__write_to_file()
            self.positions_to_write = []
            self.step_counter = 0

    def __write_to_file(self) -> None:
        self.output_file_trajectory = open(self.output_file, 'a')
        output_writer = csv.writer(self.output_file_trajectory)
        for pos in self.positions_to_write:
            lon,lat = Transformer.from_crs("EPSG:3857","EPSG:4326").transform(pos[2],pos[3])
            output_writer.writerow([self.writing_id_trajectory, f"Agent{pos[0]}", 
                                   pos[1],
                                   lat,lon,pos[4],pos[5]])
            self.writing_id_trajectory += 1
        self.output_file_trajectory.close()
        print("time: ",self.time)
        print("average locations: ",get_average_visited_locations(self))

    def __update_clock(self) -> None:
        self.time = self.time + timedelta(seconds = self.step_duration)
        if self.time > self.end_date:
            exit()