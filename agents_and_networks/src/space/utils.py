from typing import List, Tuple

import geopandas as gpd
import mesa
import numpy as np
import pyproj
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import transform
import powerlaw as powerlaw


def get_coord_matrix(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> np.ndarray:
    return np.array(
        [
            [x_min, y_min, 1.0],
            [x_min, y_max, 1.0],
            [x_max, y_min, 1.0],
            [x_max, y_max, 1.0],
        ]
    )

def power_law_exponential_cutoff(
        xmin: float, xmax:float, alpha_beta: float, k: float
) -> float:
    return powerlaw.Truncated_Power_Law(xmin = xmin,xmax = xmax,parameters=[1. + alpha_beta, 1.0 / k]).generate_random()[0]

def get_affine_transform(
    from_coord: np.ndarray, to_coord: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    A, res, rank, s = np.linalg.lstsq(from_coord, to_coord, rcond=None)

    np.testing.assert_array_almost_equal(res, np.zeros_like(res), decimal=15)
    np.testing.assert_array_almost_equal(A[:, 2], np.array([0.0, 0.0, 1.0]), decimal=15)

    # A.T = [[a, b, x_off],
    #        [d, e, y_off],
    #        [0, 0,  1  ]]
    # affine transform = [a, b, d, e, x_off, y_off]
    # For details, refer to https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.affine_transform.html
    return A.T[0, 0], A.T[0, 1], A.T[1, 0], A.T[1, 1], A.T[0, 2], A.T[1, 2]


def get_rounded_coordinate(
    float_coordinate: mesa.space.FloatCoordinate,
) -> mesa.space.Coordinate:
    return round(float_coordinate[0]), round(float_coordinate[1])


def segmented(lines: gpd.GeoSeries) -> gpd.GeoSeries:
    def _segmented(linestring: LineString) -> List[LineString]:
        return [
            LineString((start_node, end_node))
            for start_node, end_node in zip(
                linestring.coords[:-1], linestring.coords[1:]
            )
            if start_node != end_node
        ]

    return gpd.GeoSeries([segment for line in lines for segment in _segmented(line)])


# reference: https://gis.stackexchange.com/questions/367228/using-shapely-interpolate-to-evenly-re-sample-points-on-a-linestring-geodataframe
# def redistribute_vertices(geom, distance):
#     if isinstance(geom, LineString):
#
#         if (num_vert := int(round(geom.length / distance))) == 0:
#             num_vert = 1
#         return LineString(
#             [
#                 geom.interpolate(float(n) / num_vert, normalized=True)
#                 for n in range(num_vert + 1)
#             ]
#         )
#
#     elif isinstance(geom, MultiLineString):
#         parts = [redistribute_vertices(part, distance) for part in geom]
#         return type(geom)([p for p in parts if not p.is_empty])
#     else:
#         raise TypeError(
#             f"Wrong type: {type(geom)}. Must be LineString or MultiLineString."
#         )


def redistribute_vertices_new(geom, traversal_times,maxspeed,time_step) -> tuple([list[mesa.space.FloatCoordinate],list[float]]):
    traversal_times = traversal_times/time_step # Convert to fraction of timesteps for easier manipulation.
    cum_sum_traversal_times = np.append(0,np.cumsum(traversal_times))
    maxspeed = np.append(maxspeed[0],maxspeed)
    vertex_list = []
    speed_per_timestep = []
    index_old = 0
    for i in range(1,np.ceil(cum_sum_traversal_times[-1]).astype(int),1):
        index_new = np.flatnonzero(cum_sum_traversal_times>i)[0]
        distance = (i-cum_sum_traversal_times[index_new-1])/(cum_sum_traversal_times[index_new]-cum_sum_traversal_times[index_new-1])
        vertex_to_add = (geom[index_new-1].interpolate(distance,normalized=True))
        vertex_list.append(vertex_to_add)
        current_speed = ((cum_sum_traversal_times[index_old]-(i-1))*maxspeed[index_old]
                         +sum(traversal_times[index_old:index_new]*maxspeed[index_old+1:index_new+1])
                         +(i-cum_sum_traversal_times[index_new])*maxspeed[index_new]
                         )
        speed_per_timestep.append(current_speed)
        index_old = index_new
    return LineString(vertex_list),speed_per_timestep



class UnitTransformer:
    _degree2meter: pyproj.Transformer
    _meter2degree: pyproj.Transformer

    def __init__(
        self, degree_crs=pyproj.CRS("EPSG:4326"), meter_crs=pyproj.CRS("EPSG:3857")
    ):
        self._degree2meter = pyproj.Transformer.from_crs(
            degree_crs, meter_crs, always_xy=True
        )
        self._meter2degree = pyproj.Transformer.from_crs(
            meter_crs, degree_crs, always_xy=True
        )

    def degree2meter(self, geom):
        return transform(self._degree2meter.transform, geom)

    def meter2degree(self, geom):
        return transform(self._meter2degree.transform, geom)
