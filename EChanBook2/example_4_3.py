

import pandas as pd
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from johansen_test import coint_johansen
from numpy.matlib import repmat



if __name__ == "__main__":
    ###########################################################################
    # import data from MAT file
    ###########################################################################
    path = my_path('MAC')
    stocks_path = path + 'example4_1.mat'  
    etf_path = path + 'SPY_daily.csv'
    
    # get the data form sources
    stks = get_data_from_matlab(stocks_path, index='tday', columns='stocks', 
                                data='cl')
    etf = pd.read_csv(etf_path, index_col='Date')

    # daily returns
    rtn = stks.pct_change()
    # market return
    mrkt_rtn= pd.DataFrame(np.mean(rtn, axis=1))
    
    # the difference of each stocks vs the mean return
    w = (repmat(mrkt_rtn,1,len(stks.T)) - rtn)
    # divided by the absolut value of the portfolio    
    # TODO(Javier): check exactly what does this mean in the Khandani paper
    # check the math 
    w = w / repmat(pd.DataFrame(np.sum(abs(w), axis=1)), 1, len(stks.T))
    #print(sum(w, axis=1))
    print(type(w))
    print(w.shape)
    print(np.sum(w, axis=1))
    
    daily_rtn = np.sum(w[:-1] * rtn, axis=1)
    
    
    
    
    
    
    plt.plot(daily_rtn)
    plt.show()
