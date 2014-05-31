
import pandas as pd
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from johansen_test import coint_johansen
from numpy.matlib import repmat
from datetime import datetime
import cProfile
import pstats




def main():  
    ###########################################################################
    # import data from CSV file
    ###########################################################################
    path = my_path('PC')
    data = pd.read_csv(path + 'CAD_AUD.csv'  , index_col='Date')
    
    CAD = 1 / pd.DataFrame(data['USDCAD'])
    AUD = pd.DataFrame(data['AUDUSD'], index=data.index)
    y = AUD.join(CAD)

    # strategy parameters
    lookback = 20
    trainlen = 250
    
    hedgeratio = pd.DataFrame(np.zeros([len(data),2]), columns=y.columns, index=y.index)
    numunits = pd.DataFrame(np.zeros([len(data),1]), columns=['numunits'], index=y.index)
    
    
    for t in range(trainlen+1,len(data)):
        results = coint_johansen(log(data[t-trainlen:t]), 0, 1 , print_on_console=False)
        hedgeratio.iloc[t] = results.evec[0,0], results.evec[1,0]
        # we apply the t+0 hedgeratio to all the data in the lookback period
        # for calculation of the yport
        yport = np.sum(y[t-lookback:t] * repmat(hedgeratio.iloc[t], lookback, 1), axis=1)
        ma = np.mean(yport)
        std = np.std(yport)
        numunits.iloc[t] = -(yport[-1] - ma) / std
    
    positions = repmat(numunits, 1, 2) * hedgeratio * y 

    pnl = np.sum(positions.shift(1) * y.pct_change(1), axis=1)
    rtn = pnl / np.sum(np.abs(positions.shift(1)), axis=1)
    rtn= rtn[trainlen+2:]
    #rtn.to_csv(path+'rtn.csv')
    cumret = pd.DataFrame(np.cumprod((1+rtn))-1, index=rtn.index)
    #print(rtn.head())
    print(cumret.head())
    print(cumret.tail())
    #cumret = cumret.fillna(method='pad')
    
    # compute performance statistics
    sharpe = (np.sqrt(252)*np.mean(rtn)) / np.std(rtn)
    APR = np.prod(1+rtn)**(252/len(rtn))-1
    
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
    ax.plot(cumret)
    ax.set_title('C')
    ax.set_xlabel('Data points')
    ax.set_ylabel('acumm rtn')
    ax.text(1000, -0.05, 'Sharpe: {:.4}'.format(sharpe))
    ax.text(1000, 0, 'APR: {:.4%}'.format(APR))
    
    plt.show()



#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
#with PyCallGraph(output=GraphvizOutput()):
#    strategy()

if __name__ == "__main__":
    cProfile.run('import example_5_1; example_5_1.main()', 'profile.tmp')
    p = pstats.Stats('profile.tmp')
    p.sort_stats('cumulative').print_stats(10)