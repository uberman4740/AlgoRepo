from johansen_test import coint_johansen
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from numpy.matlib import repmat
import numpy as np

# Implementation of the second part of example 3.1 in Ernest Chang's 
# book Algorithmic trading : winning strategies and their rationale.
#
# log (Price) Spread

if __name__ == "__main__":
   
    #import data from CSV file
    root_path = 'C:/Users/javgar119/Documents/Python/Data/'
    # the paths
    # MAC: '/Users/Javi/Documents/MarketData/'
    # WIN: 'C:/Users/javgar119/Documents/Python/Data/'
    filename = 'GLD_USO_daily2.csv'
    full_path = root_path + filename
    data = pd.read_csv(full_path, index_col='Date')
    lookback = 20
    data = data[lookback-1:]
    y_ticket = 'USO'
    x_ticket = 'GLD'
    
    y = data[y_ticket]
    x = data[x_ticket]
    
    # ratio USO/GLD
    ratio = y/x
    # lookback period for calculating the dynamically changing

    # remove some data to have the same dataset
    
    moving_mean = pd.rolling_mean(ratio, window=lookback) 
    moving_std = pd.rolling_std(ratio, window=lookback) 
    # calculate the number of units for the strategy in the form

    # the number of units of the syntetic portfolio is given by the
    # negative value of z-score
    numunits = pd.DataFrame(-(ratio - moving_mean) / moving_std)
    

    # compute the $ position for each asset
    AA = repmat(numunits,1,2)
    BB = repmat([-1,1],len(numunits),1)
    position = multiply(data, multiply(AA,BB))
      
    # compute the daily pnl in $$
    pnl = sum((multiply(position[:-1], divide(diff(data,axis = 0), data[:-1]))),1)
    
    # gross market value of portfolio
    mrk_val = pd.DataFrame.sum(abs(position), axis=1)
    mrk_val = mrk_val[lookback-1:-1]

    # return is P&L divided by gross market value of portfolio
    rtn = pnl / mrk_val

    # compute performance statistics
    sharpe = (np.sqrt(252)*np.mean(rtn)) / np.std(rtn)
    APR = np.prod(1+rtn)**(252/len(rtn))-1
    
    ##################################################
    # print the results
    ##################################################
    print('Price spread Sharpe: {:.4}'.format(sharpe))
    print('Price Spread APR: {:.4%}'.format(APR))
    
    
    #*************************************************
    # plotting the chart
    #*************************************************    
    #plot of numunits
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cumsum(rtn))
    ax.set_title('{}-{} Price Ratio Acum Return'.format(x_ticket, y_ticket))
    ax.set_xlabel('Data points')
    ax.set_ylabel('acumm rtn')
    ax.text(1000, 0, 'Sharpe: {:.4}'.format(sharpe))
    ax.text(1000, -0.06, 'APR: {:.4%}'.format(APR))
    
   
    
    plt.show()