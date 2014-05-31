

import pandas as pd
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat



if __name__ == "__main__":
    ###########################################################################
    # import data from MAT file
    ###########################################################################
    path = my_path('MAC')
    stocks_path = path + 'example4_1.mat'
    #'example4_1.mat'
    #'example4_3.csv'  
    # get the data form sources
    cl = get_data_from_matlab(stocks_path, index='tday', columns='stocks', 
                                data='cl')
    op = get_data_from_matlab(stocks_path, index='tday', columns='stocks', 
                                data='op')
    #stks = pd.read_csv(stocks_path, index_col='Date')
    
    
    ###########################################################################
    # Cross-Sectional Mean Reversion Strategy
    ###########################################################################
    
    # daily returns
    #rtn = stks.pct_change()

    
    rtn = (op - cl.shift(1)) / cl.shift(1) 


    # market return
    mrkt_rtn= pd.DataFrame(np.mean(rtn, axis=1)) 
    
    # the difference of each stocks vs the mean return
    w = -(rtn - repmat(mrkt_rtn,1,len(cl.T)))
    # divided byabsolute value of the portfolio, portfolio always is 1 
    w = w / repmat(pd.DataFrame(np.sum(abs(w), axis=1)), 1, len(cl.T))
    
    # divided by the absolut value of the portfolio    
    daily_rtn = np.sum(w * ((cl-op) / op), axis=1)

    # compute performance statistics
    sharpe = (np.sqrt(252)*np.mean(daily_rtn)) / np.std(daily_rtn)
    APR = np.prod(1+daily_rtn)**(252/len(daily_rtn))-1
    
    ##################################################
    # print the results
    ##################################################
    print('Sharpe: {:.4}'.format(sharpe))
    print('APR: {:.4%}'.format(APR))
    
    ###########################################################################
    # plotting the chart
    ###########################################################################

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.cumprod(1+daily_rtn)-1)
    ax.set_title('Cross-Sectional Mean Reversion')
    ax.set_xlabel('Data points')
    ax.set_ylabel('acumm rtn')
    ax.text(1000, 1, 'Sharpe: {:.4}'.format(sharpe))
    ax.text(1000, 0, 'APR: {:.4%}'.format(APR))
    
    plt.show()
