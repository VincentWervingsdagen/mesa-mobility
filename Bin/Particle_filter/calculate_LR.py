import numpy as np

from scripts.Particle_filter.Particle import ParticleFilter
import pandas as pd
import pyproj


class CalculateLR:
    path_constructor: ParticleFilter
    observations: pd.DataFrame
    transformer: pyproj.Transformer

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
                 method = 1):

        self.transformer = pyproj.Transformer.from_crs(crs_from=data_crs,crs_to=graph_crs,always_xy=True)
        self.load_observations_from_file(observation_file)
        self.select_observation_triple()
        # ParticleFilter(roadnetwork_file=roadnetwork_file,
        #                coverage_file_path=coverage_file_path,
        #                observation_file=observation_file,
        #                output_file=output_file,
        #                walking_allowed=walking_allowed,
        #                bounding_box=bounding_box,
        #                N=N,
        #                data_crs=data_crs,
        #                graph_crs=graph_crs,
        #                method=method)

    def load_observations_from_file(self, dataframe_file: str) -> None:
        self.observations = pd.read_csv(dataframe_file, )
        self.observations = self.observations.set_index('cellinfo.id', drop=False)
        self.observations['timestamp'] = pd.to_datetime(self.observations['timestamp'], format='%Y-%m-%d %H:%M:%S')
        self.observations['x'], self.observations['y'] = self.transformer.transform(
            self.observations['cellinfo.wgs84.lon'], self.observations['cellinfo.wgs84.lat'])
        self.observations = self.observations.sort_values('timestamp')

    def select_observation_triple(self):
        print(self.observations.head(10))
        list_index = np.cumsum(self.observations['device']=='0_1')
        list_index2 = np.cumsum(self.observations['device'] == '0_2')
        print(list_index)
        print(list_index2)



        # different_device = self.observations['device'].ne(self.observations['device'].shift(-1))
        # possible_triples = (different_device.rolling(window=2).sum() == 2) & (self.observations['device'] == '0_1')
        # indexes = self.observations[possible_triples]



        # next_position = self.observations.groupby(['owner','device'])[['cellinfo.wgs84.lon','cellinfo.wgs84.lat']].shift()
        # same_cell_tower = self.observations[['cellinfo.wgs84.lon','cellinfo.wgs84.lat']].eq(next_position).mask(next_position.isna(),'')
        # print(self.observations[['cellinfo.wgs84.lon','cellinfo.wgs84.lat']])
        # print(next_position)
        # print(same_cell_tower)



