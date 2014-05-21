
import pandas as pd
from numpy import diag, eye, zeros, pi, exp, dot, sum, tile, linalg, array
from numpy import log as logg
from numpy.linalg import inv, det
from numpy.random import randn
import numpy as np
import random
import matplotlib.pyplot as plt





# Implementation of a Kalman filter for time series analysis.
# Ideas adopted from http://hal.archives-ouvertes.fr/docs/00/43/38/86/PDF/Laaraiedh_PythonPapers_Kalman.pdf

# Algorithm plots both the observed price and the signal produced
# by the Kalman filter for assets in the universe.

def initialize(x, y):
    # Set total number of securities
    
    total_securites = 2

    
    X = np.zeros((total_securites*2,1))
    P = np.diag(np.zeros(total_securites*2))   
    A = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0,0]])    
    Q = eye(X.shape[0])
    B = eye(X.shape[0])
    U = zeros((X.shape[0],1))
    #Y = array([[0], [0]])
    Y = array(x)
    H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = eye(Y.shape[0])

    # Apply the Kalman filter
    # Prediction step 
    N_iter = 50 

    for i in range(N_iter):
        
        (X, P) = kf_predict(X, P, A, Q, B, U)
        # Update step
        (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)  
        # Record observed price data
        
      
       
    print(X)
    
# PREDICTION STEP

# X is the mean state estimate of the previous step (k-1)
# P is the state covariance matrix of the previous step (k-1)
# A is the transition nxn matrix
# Q is the process noise covariance matrix
# B is the input effect matrix
# U is the control input

def kf_predict(X, P, A, Q, B, U): 
 X = dot(A, X) + dot(B, U) 
 P = dot(A, dot(P, A.T)) + Q 
 return(X,P) 


# UPDATE STEP

# At time step k, this update computes the posterior mean X and 
# covariance P of the system state given a new measurement Y. 

# Y is the measurement vector
# H is the measurement matrix
# R is the measurement covariance matrix
# K is the Kalman Gain matrix
# IM is the mean of the predictive distribution of Y
# IS is the covariance of the predictive mean of Y
# LH is the predictive probability (lieklihood) of measurement

def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X,P,K,IM,IS,LH)
    
def gauss_pdf(X, M, S):
    if M.shape[1] == 1:
        DX = X - tile(M, X.shape[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * logg(2 * pi) + 0.5 * logg(det(S))
        P = exp(-E)
    elif X.shape[1] == 1:
        DX = tile(X, M.shape[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * logg(2 * pi) + 0.5 * logg(det(S))
        P = exp(-E)
    else:
        DX = X-M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * logg(2 * pi) + 0.5 * logg(det(S))
        P = exp(-E)
    return (P[0],E[0])
    


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
       #import data from CSV file
    root_path = 'C:/Users/javgar119/Documents/Python/Data/'
    # the paths
    # MAC: '/Users/Javi/Documents/MarketData/'
    # WIN: 'C:/Users/javgar119/Documents/Python/Data/'
    filename = 'EWC_EWA_daily.csv'   #version 2 is bloomberg data
    full_path = root_path + filename
    data = pd.read_csv(full_path, index_col='Date')
   
    y_ticket = 'EWC'
    x_ticket = 'EWA'
    
    y = data[y_ticket]
    x = data[x_ticket]
    
    initialize(x=x, y=y)