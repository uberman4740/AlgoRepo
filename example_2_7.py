from johansen_test import coint_johansen
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from functions import *

from numpy.matlib import repmat

#from numpy import *
#from numpy.linalg import *


if __name__ == "__main__":
   
    #import data from CSV file
    root_path = 'C:/Users/javgar119/Documents/Python/Data/'
    # the paths
    # MAC: '/Users/Javi/Documents/MarketData/'
    # WIN: 'C:/Users/javgar119/Documents/Python/Data'
    filename_x = 'EWC_EWA_IGE_daily.csv'
    #filename_y = 'ECOPETROL_ADR.csv'
    full_path_x = root_path + filename_x
    #full_path_y = root_path + filename_y
    data = pd.read_csv(full_path_x, index_col='Date')
    #create a series with the data range asked
    #start_date = '2006-01-03'
    #end_date = '2012-04-17'
    #data = subset_dataframe(data, start_date, end_date)
    #print('data import is {}'.format(str(len(data))))
    #print(data.head(10))
    #print(data.tail(10))
    #johansen test with non-zero offset but zero drift, and with the lag k=1.
    results = coint_johansen(data, 0, 1)
    # those are the weigths of the portfolio
    # the first eigenvector because it shows the strongest cointegration relationship
    
    w = results.evec[:, 0]
 
    #print(w)
    
    # (net) market value of portfolio
    # this is the syntetic asset we are going to trade. A freshly new mean reverting serie
    # compose of the three assets in proportions given by the eigenvector
    yport = pd.DataFrame.sum(w*data, axis=1)



    # LINEAR STRATEGY
    # A linear trading strategy means that the number of units or shares of a
    # unit portfolio we own is proportional to the negative z-score of the
    # price series of that portfolio.
    
    lookback = int(half_life(yport))



    moving_mean = pd.rolling_mean(yport, window=lookback)
    moving_std = pd.rolling_std(yport,window=lookback)

    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])
    
    AA = repmat(numunits,1,3)
    BB = multiply(repmat(w,len(data),1),data)
    position = multiply(AA, BB)

    
    pnl = sum(divide(multiply(position[:-1],diff(data,axis = 0)), data[:-1]),1)
    
 
    
    plt.plot(cumsum(pnl[25:]))
    plt.show()