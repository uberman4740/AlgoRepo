'''
==================================
Kalman Filter tracking a sine wave
==================================

This example shows how to use the Kalman Filter for state estimation.

In this example, we generate a fake target trajectory using a sine wave.
Instead of observing those positions exactly, we observe the position plus some
random noise.  We then use a Kalman Filter to estimate the velocity of the
system as well.

The figure drawn illustrates the observations, and the position and velocity
estimates predicted by the Kalman Smoother.
'''
import numpy as np
import pylab as pl
import pandas as pd
from pykalman import KalmanFilter
from pykalman.datasets import load_robot

rnd = np.random.RandomState(0)

# generate a noisy sine wave to act as our fake observations



#import data from CSV file
root_path = 'C:/Users/javgar119/Documents/Python/Data/'
filename = 'EWC_EWA_daily.csv'   #version 2 is bloomberg data
full_path = root_path + filename
data = pd.read_csv(full_path, index_col='Date')
   
y_ticket = 'EWC'
x_ticket = 'EWA'
y = data[y_ticket]
x = data[x_ticket]

x = x[:100]

observations=np.asarray(x)

n_timesteps = len(x)
x = np.linspace(0, n_timesteps, n_timesteps)

#transition_matrices
#β(t) = β(t − 1) + ω(t − 1), (“State transition”) 
#where ω is also a Gaussian noise but with covariance Vω. In other words, the
#state transition model here is just the identity matrix.
A = np.array([[1, 0], [0, 1]])
#observation_matrices



# create a Kalman Filter by hinting at the size of the state and observation
# space.  If you already have good guesses for the initial parameters, put them
# in here.  The Kalman Filter will try to learn the values of all variables.
kf = KalmanFilter(A, transition_covariance=0.01 * np.eye(2))


# You can use the Kalman Filter immediately without fitting, but its estimates
# may not be as good as if you fit first.
states_pred = kf.em(observations).smooth(observations)[0]

#(filtered_state_means, filtered_state_covariances) = kf.filter(observations)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(observations)




print('fitted model: {0}'.format(kf))

# Plot lines for the observations without noise, the estimated position of the
# target before fitting, and the estimated position after fitting.

pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(x, observations, marker='x', color='b',
                         label='observations')
position_line = pl.plot(x, states_pred[:, 0],
                        linestyle='-', color='r',
                        label='position est.')
#velocity_line = pl.plot(x, states_pred[:, 1],
#                        linestyle='-', marker='o', color='g',
#                        label='velocity est.')
pl.legend(loc='lower right')
pl.xlim(xmin=0, xmax=x.max())
pl.xlabel('time')
pl.show()