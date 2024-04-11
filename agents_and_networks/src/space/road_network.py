from __future__ import annotations

import pickle
import numpy as np
import geopandas as gpd
import mesa
import momepy
import networkx as nx
import pandas as pd
import pyproj
from sklearn.neighbors import KDTree
from shapely import LineString
from src.space.utils import segmented,UnitTransformer
from random import choice
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def count_line_segments(line):
    return len(line.coords)-1


class RoadNetwork:
    _nx_graph: nx.Graph
    _kd_tree: KDTree
    _crs: pyproj.CRS

    def __init__(self, lines: gpd.GeoSeries,maxspeed:pd.Series):
        segmented_lines = gpd.GeoDataFrame(geometry=segmented(lines))
        segmented_maxspeed = np.repeat(maxspeed.values, lines.apply(count_line_segments))/3.6  # in meter per second
        traversal_time = segmented_lines.length/segmented_maxspeed  # in seconds
        segmented_lines['maxspeed'] = segmented_maxspeed
        segmented_lines['traversal_time'] = traversal_time
        G = momepy.gdf_to_nx(segmented_lines, approach="primal", length='length', multigraph=False)
        self.nx_graph = G.subgraph(max(nx.connected_components(G), key=len))
        self.crs = lines.crs


    @property
    def nx_graph(self) -> nx.Graph:
        return self._nx_graph

    @nx_graph.setter
    def nx_graph(self, nx_graph) -> None:
        self._nx_graph = nx_graph
        self._kd_tree = KDTree(nx_graph.nodes)

    @property
    def crs(self) -> pyproj.CRS:
        return self._crs

    @crs.setter
    def crs(self, crs) -> None:
        self._crs = crs

    def get_nearest_node(
        self, float_pos: mesa.space.FloatCoordinate
    ) -> mesa.space.FloatCoordinate:
        node_index = self._kd_tree.query([float_pos], k=1, return_distance=False)
        node_pos = self._kd_tree.get_arrays()[0][node_index[0, 0]]
        return tuple(node_pos)

    def get_shortest_path(
        self, source: mesa.space.FloatCoordinate, target: mesa.space.FloatCoordinate
    ) -> list[mesa.space.FloatCoordinate]:
        from_node_pos = self.get_nearest_node(source)
        to_node_pos = self.get_nearest_node(target)
        path = nx.astar_path(self.nx_graph, from_node_pos, to_node_pos, weight="traversal_time")
        return path

    def get_maxspeed(self,path: list[LineString])->np.array[float]:
        return np.array([self.nx_graph[path[i].coords[0]][path[i].coords[1]]['maxspeed'] for i in range(len(path))])

    def get_traversal_times(self,path: list[LineString])->np.array[float]:
        return np.array([self.nx_graph[path[i].coords[0]][path[i].coords[1]]['traversal_time'] for i in range(len(path))])

    def get_length_shortest_path(
        self, source: mesa.space.FloatCoordinate, target: mesa.space.FloatCoordinate
    ) -> int:
        from_node_pos = self.get_nearest_node(source)
        to_node_pos = self.get_nearest_node(target)
        length = nx.astar_path_length(self.nx_graph, from_node_pos, to_node_pos, weight="traversal_time")
        return length


class NetherlandsWalkway(RoadNetwork):
    _path_select_cache: dict[
        tuple[mesa.space.FloatCoordinate, mesa.space.FloatCoordinate],
        list[mesa.space.FloatCoordinate],
    ]

    def __init__(self, lines,maxspeed) -> None:
        super().__init__(lines,maxspeed)
        self._path_cache_result = os.path.join(script_dir, '..','..', 'outputs', 'path_cache_result.pkl')
        try:
            with open(self._path_cache_result, "rb") as cached_result:
                self._path_select_cache = pickle.load(cached_result)
        except FileNotFoundError:
            self._path_select_cache = {}

    def cache_path(
        self,
        source: mesa.space.FloatCoordinate,
        target: mesa.space.FloatCoordinate,
        path: list[mesa.space.FloatCoordinate],
    ) -> None:
        # print(f"caching path... current number of cached paths:
        # {len(self._path_select_cache)}")
        self._path_select_cache[(source, target)] = path
        self._path_select_cache[(target, source)] = list(reversed(path))
        with open(self._path_cache_result, "wb") as cached_result:
            pickle.dump(self._path_select_cache, cached_result)

    def get_cached_path(
        self, source: mesa.space.FloatCoordinate, target: mesa.space.FloatCoordinate
    ) -> list[mesa.space.FloatCoordinate] | None:
        return self._path_select_cache.get((source, target), None)
