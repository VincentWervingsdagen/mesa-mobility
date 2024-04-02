import geopy.distance
import numpy as np
import pandas as pd
from scipy import stats
from geopy.distance import distance
import datetime

import os

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
    particles[:, 2] = np.clip(particles[:, 2] + stats.norm(loc=0, scale=5).rvs(N), 0, 100)
    particles[:, 3] = (particles[:, 3] + stats.norm(loc=0, scale=90).rvs(N)) % 360
    for particle in particles:
        particle[0],particle[1] = distance(dt*particle[2]).destination((particle[0],particle[1]),bearing=particle[3])[0:2]
    return particles

def update(particles,data):
    N=len(particles)
    weights = np.ones(N)
    for i in range(0,N):
        weights[i] += distance((particles[i,0],particles[i,1]),(data.iloc[0,0],data.iloc[0,1])).meters
    weights /= sum(weights)  # normalize
    return weights

def estimate(particles, weights):
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
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
    time_old = datetime.strptime('2023-05-01 00:00:00','%Y-%m-%d %H:%M:%S')
    estimate_particle = np.empty((len(data),2))
    for i in range(len(data)):
        time_new = data['timestamp'].iloc[i]
        particles = predict(particles,datetime.timedelta(time_new,time_old))
        weights = update(particles, data[['cellinfo.wgs84.lat','cellinfo.wgs84.lon']].iloc(i))
        estimate_particle[i] = estimate(particles,weights)
        particles = simple_resample(particles,weights)
        time_old = time_new

    return estimate_particle

data = pd.read_csv(open(os.path.join(script_dir, '..','..', 'outputs', 'trajectories', 'output_cell.csv'))).drop(['owner','id','device','cell'],axis=1)
print(data.head())

print(simulation(51.9181,4.4739,5,data))
