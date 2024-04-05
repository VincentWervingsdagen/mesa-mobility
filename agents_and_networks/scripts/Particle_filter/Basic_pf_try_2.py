import numpy as np
import pandas as pd
from scipy import stats
from geopy.distance import distance
import datetime
import shapely
import os
import pyproj

script_dir = os.path.dirname(os.path.abspath(__file__))


def create_particles(x, y, N):
    particles = np.empty((N, 4))
    particles[:, 0] = stats.norm(loc=x, scale=0.03).rvs(N)
    particles[:, 1] = stats.norm(loc=y, scale=0.03).rvs(N)
    particles[:, 2] = stats.uniform(0,50).rvs(N)
    particles[:, 3] = stats.uniform(0, 360).rvs(N)
    return particles


def predict(particles, dt=1.):
    N = len(particles)
    particles[:, 2] = np.clip(particles[:, 2] + stats.norm(loc=0, scale=20).rvs(N), 0, 100)
    bearing = np.concatenate([stats.norm(loc=-90,scale=10).rvs(N),stats.norm(loc=0,scale=10).rvs(N),stats.norm(loc=90,scale=10).rvs(N)])
    np.random.shuffle(bearing)
    particles[:, 3] = (particles[:, 3] + bearing[0:N]) % 360
    for particle in particles:
        particle[0],particle[1] = distance(dt*particle[2]).destination((particle[0],particle[1]),bearing=particle[3])[0:2]
    return particles


def update(particles,data):
    N=len(particles)
    weights = np.ones(N)/1000
    wgs84 = pyproj.CRS("EPSG:4326")
    utm = pyproj.CRS("EPSG:32631")
    transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
    triangle_point1 = transformer.transform(*distance(2).destination((data[0],data[1]),bearing=data[2]+60)[0:2])
    triangle_point2 = transformer.transform(*distance(2).destination((data[0], data[1]), bearing=data[2] - 60)[0:2])
    triangle = shapely.geometry.Polygon([transformer.transform(data[0],data[1]),triangle_point1,triangle_point2])
    for i in range(0,N):
        weights[i] += min(1,1/(0.01+shapely.geometry.Point(transformer.transform(particles[i,0],particles[i,1])).distance(triangle)/1000))
    weights /= sum(weights)  # normalize
    return weights


def estimate(particles, weights,data):
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    print(distance((mean[0],mean[1]),(data[0],data[1])))
    return mean


def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, stats.uniform(0,1).rvs(N))
    # resample according to indexes
    return particles[indexes]


def simulation(x,y,N,data):
    particles = create_particles(x,y,N=N)
    time_old = datetime.datetime.strptime('2023-05-01 00:00:00','%Y-%m-%d %H:%M:%S')
    estimate_particle = np.empty((len(data),2))
    for i in range(len(data)):
        time_new = datetime.datetime.strptime(data['timestamp'][i],'%Y-%m-%d %H:%M:%S')
        time_difference = (time_new-time_old)/datetime.timedelta(hours=1)
        particles = predict(particles,time_difference)
        weights = update(particles, data[['cellinfo.wgs84.lat','cellinfo.wgs84.lon','cellinfo.azimuth_degrees']].iloc[i])
        estimate_particle[i] = estimate(particles,weights,data[['cellinfo.wgs84.lat','cellinfo.wgs84.lon']].iloc[i])
        particles = simple_resample(particles,weights)
        time_old = time_new
    return estimate_particle


data = pd.read_csv(open(os.path.join(script_dir, '..','..', 'outputs', 'trajectories', 'output_cell.csv')))

print(data.columns)

data[['estimate_lat','estimate_lon']] = simulation(51.9181,4.4739,1000,data)

data.to_csv(os.path.join(script_dir, '..','..', 'outputs', 'trajectories', 'output_cell.csv'))
