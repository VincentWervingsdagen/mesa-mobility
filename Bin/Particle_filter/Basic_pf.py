import numpy as np
import pandas as pd
from scipy import stats
from particles import state_space_models as ssm
from particles import distributions as dists
import particles
from geopy.distance import distance

import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the motion model
class movingParticle(ssm.StateSpaceModel):
    default_params = {'sigmaX': 1,
                      'sigmaY': 5,
                      'x0': np.array([51.9181,4.4739, 0, 0])
                      }
    def PX0(self):
        return dists.IndepProd(dists.Dirac(loc=self.x0[0]),
                               dists.Dirac(loc=self.x0[1]),
                               dists.Uniform(-50,50),
                               dists.Uniform(0,360),
                               )

    def PX(self, t, xp):
        return dists.IndepProd(dists.Dirac(distance(t*xp[2]/1000).destination((xp[0],xp[1]),bearing=xp[3])[0]),
                           dists.Dirac(distance(t*xp[2]/1000).destination((xp[0],xp[1]),bearing=xp[3])[1]),
                           dists.Normal(loc=max(-100,min(100,xp[:, 2])), scale=self.sigmaX),
                           dists.Normal(xp[:, 3]%360, scale=self.sigmaY),
                           )

    def PY(self, t, xp, x):
        lat, lon, _ = distance(stats.norm(2, 0.5).rvs(1)).destination((x[:,0], x[:,1]), bearing=stats.uniform(0, 360).rvs(1))
        return dists.Dirac(loc=lat)

data = pd.read_csv(open(os.path.join(script_dir, '..','..', 'outputs', 'trajectories', 'output_cell.csv')))

# Create particle filter
mymodel = movingParticle()
fk_model = ssm.Bootstrap(ssm=mymodel, data=data[['cellinfo.wgs84.lat','cellinfo.wgs84.lon']])

pf = particles.SMC(fk=fk_model, N=100, resampling='stratified', store_history=True)  # the algorithm
pf.run()  # actual computation

# Get estimated position
estimated_x = np.mean(pf.X, axis=0)
estimated_y = np.mean(pf.X, axis=1)
print("Estimated position:", estimated_x, estimated_y)