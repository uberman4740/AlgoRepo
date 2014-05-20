from johansen_test import coint_johansen
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from numpy.matlib import repmat
import numpy as np

# Implementation of the second part of example 3.1 in Ernest Chang's 
# book Algorithmic trading : winning strategies and their rationale.
#
# log(Price) Spread

if __name__ == "__main__":
   
    #import data from CSV file
    root_path = 'C:/Users/javgar119/Documents/Python/Data/'
    # the paths
    # MAC: '/Users/Javi/Documents/MarketData/'
    # WIN: 'C:/Users/javgar119/Documents/Python/Data'
    filename = 'GLD_USO_daily.csv'
    full_path = root_path + filename
    data = pd.read_csv(full_path, index_col='Date')
   
    y_ticket = 'USO'
    x_ticket = 'GLD'
    
    y = data[y_ticket]
    x = data[x_ticket]
    

    # lookback period for calculating the dynamically changing
    lookback = 20
    modelo = pd.ols(y=np.log(y), x=np.log(x), window_type='rolling', window=lookback)
    data = data[lookback-1:]
    betas = modelo.beta
    
    # calculate the number of units for the strategy in the form
    # y-beta*x
    yport = pd.DataFrame(np.log(data[y_ticket]) - (betas['x'] * np.log(data[x_ticket])))
    
    moving_mean = pd.rolling_mean(yport, window=lookback)
    moving_std = pd.rolling_std(yport,window=lookback)
    # the number of units of the syntetic portfolio is given by the
    # negative value of z-score
    numunits = pd.DataFrame(-(yport - moving_mean) / moving_std)

    # compute the $ position for each asset
    AA = pd.DataFrame(repmat(numunits,1,2))
    BB = pd.DataFrame(-betas['x'])
    BB['ones'] = np.ones((len(betas)))
    position = multiply(multiply(AA, BB), data)

    # compute the daily pnl in $$
    pnl = sum(multiply(position[:-1], divide(diff(data,axis = 0), data[:-1])),1)
    
        
    # gross market value of portfolio
    mrk_val = pd.DataFrame.sum(abs(position), axis=1)
    mrk_val = mrk_val[:-1]
   
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
    ax.set_title('{}-{} log(Price) Spread Acum Return'.format(x_ticket, y_ticket))
    ax.set_xlabel('Data points')
    ax.set_ylabel('acumm rtn')
    ax.text(1000, 0, 'Sharpe: {:.4}'.format(sharpe))
    ax.text(1000, -0.03, 'APR: {:.4%}'.format(APR))
    
   
    
    plt.show()