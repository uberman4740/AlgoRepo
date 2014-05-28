
import pandas as pd
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from johansen_test import coint_johansen
from numpy.matlib import repmat


class strategy (object):
    def __init__(self, name):
        self.name = name
        
      

if __name__ == "__main__":
    ################################################################
    # import data from MAT file
    ################################################################
    path = my_path('MAC')
    stocks_path = path + 'example4_1.mat'  
    etf_path = path + 'SPY_daily.csv'
    
    # get the data form sources
    stks = get_data_from_matlab(stocks_path, index='tday', columns='stocks', data='cl')
    etf = pd.read_csv(etf_path, index_col='Date')

    
    #prepare the training and testing sets
    starts = '2007-01-01'
    ends = '2008-01-01'
    stks_train = pd.DataFrame(stks[starts:ends], columns=stks.columns)
    etf_train = pd.DataFrame(etf[starts:ends], columns=etf.columns) 
    stks_test = pd.DataFrame(stks[ends:], columns=stks.columns)
    etf_test = pd.DataFrame(etf[ends:], columns=etf.columns)
    
    
    ###########################################################################
    # Find the cointegrating portfolio
    ###########################################################################
    
    
    # johansen cointegration test for each stock vs SPY
    stks_list = list(stks.columns) 
    isCoint = pd.DataFrame(np.zeros([1,len(stks_list)], dtype=bool), columns=stks_list, 
                           index=['isCoint'])
    # set the confidence level
    confidence = 0.90
    if confidence==0.95:
        x=1
    elif confidence==0.99: 
        x=2
    else: 
        x=0
 
    for col in isCoint:
        # join the SPY to each of the stocks members in a Nx2 dataframe
        # clean for missing values
        y2 = pd.DataFrame(stks_train[col]).join(etf_train).dropna()  
        # run johansen test for each of the N stocks vs SPY.
        if len(y2) > 250: 
            results = coint_johansen(y2, 0, 1, print_on_console=False)
            if results.lr1[0] > results.cvt[0,x]:
                isCoint[col] = True
 
 
    print('Johansen Test results for each stock vs SPY')
    print('--------------------------------------------------')
    print('Universe is {} stocks in the index'.format(len(stks_list)))
    print('Johansen Test over {} data points'.format(len(stks_train)))
    print('There are {} stocks that cointegrate with {:.2%} confidence'.format(np.count_nonzero(isCoint), 
                                                                               confidence))           
    print('--------------------------------------------------')
    print()
    print()
   
    # capital allocation
    yN= repmat(isCoint, len(stks_train), 1) * stks_train
    yN = yN.replace(0, np.NaN)
    # the net market value of the long only portfolio 
    # We are using log price in this second test because we expect to rebalance 
    # this portfolio every day so that the capital on each stock is constant.
    log_mrk_val = pd.DataFrame(np.sum(log(yN), axis=1), columns=['log_mrk_val'])
    
    # Confirm that the portfolio cointegrates with SPY
    ytest = log_mrk_val.join(log(etf_train))
    print('Johansen Test results\nfor log value of portfolio vs log(SPY)')
    results = coint_johansen(ytest, 0, 1, print_on_console=True)
    

   
    ###########################################################################
    # Apply linear mean-reversion model on test set
    ###########################################################################
    yNplus = (repmat(isCoint, len(stks_test), 1) * stks_test).join(etf_test)
    yNplus = yNplus.replace(0, np.NaN)
    
    
    weights = np.hstack((repmat(results.evec[0, 0], len(stks_test), len(isCoint.T)),
                         repmat(results.evec[1, 0], len(stks_test), 1)))
    
    
    #Log market value of long-short portfolio
    log_mrk_val = np.sum(log(yNplus) * weights, axis=1)
    

    lookback = 5
    moving_mean = pd.rolling_mean(log_mrk_val, window=lookback)
    moving_std = pd.rolling_std(log_mrk_val,window=lookback)
    
    # capital invested in portfolio in dollars.
    numunits = pd.DataFrame(-(log_mrk_val - moving_mean) / moving_std)
    
    # positions is the dollar capital in each stock or ETF
    positions = pd.DataFrame(repmat(numunits, 1, len(weights.T)) *  weights)
    # the number of each stock held in the portolio
    stks_position = 1
    #  daily P&L of the strategy

    
    pnl = np.sum(multiply(positions[:-1], diff(log(yNplus), axis = 0)), axis=1)
    gross_mrk_val = np.sum(positions, axis=1)
    rtn = pnl / gross_mrk_val
    
    #print('yNplus: {} / {}'.format(yNplus.shape, type(yNplus)))
    #print('positions: {} / {}'.format(positions.shape, type(positions)))
    #print('pnl: {} / {}'.format(pnl.shape, type(pnl)))
    
   
    
    # cumulative return and smoothing of series for plotting
    #acum_rtn = cumprod(1+rtn)-1
    #acum_rtn = acum_rtn.fillna(method='pad')
    # compute performance statistics
    sharpe = (np.sqrt(252)*np.mean(rtn)) / np.std(rtn)
    APR = np.prod(1+rtn)**(252/len(rtn))-1
    
    ################################################################
    # print the results
    ################################################################
    print('Sharpe: {:.4}'.format(sharpe))
    print('APR: {:.4%}'.format(APR))
    
    ################################################################
    # plotting the chart
    ################################################################
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cumprod(1+rtn)-1)
    ax.set_title('Arbitrage between SPY and Its Component Stocks')
    ax.set_xlabel('Data points')
    ax.set_ylabel('cum rtn')
    ax.text(800, 0, 'Sharpe: {:.4}'.format(sharpe))
    ax.text(800, 0.01, 'APR: {:.4%}'.format(APR))
    
    
    
    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111)
    #ax2.plot(e, color='blue')
    #ax2.plot(sqrt_Q, color='black')
    #ax2.plot(-sqrt_Q, color='black')

    #ax2.plot(sqrt_Q)
    #ax2.plot(-sqrt_Q)
    #ax2.set_title('error vs sqrt(variance prediction)')
    #ax2.set_xlabel('Data points')
    #ax2.set_ylabel('# std')
    #ax2.set_ylim(-2, 2)
    
    
    plt.show()
    # chart example with prettyplotlib
    #fig, ax = plt.subplots(1)
    #y = acum_rtn
    #x = acum_rtn.index
    #i = 'Price Ratio CumRet\n(Using Kalman filter)'
    #ppl.plot(x, y, label=str(i), linewidth=0.75)
    #ppl.legend(ax)
    #fig.savefig('kelman strat.png')


   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
  

    
    
    
    