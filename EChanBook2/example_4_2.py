
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
    path = my_path('PC')
    stocks_path = path + 'example4_1.mat'  
    etf_path = path + 'SPY_daily2.csv'
    
    # get the data form sources
    stks = get_data_from_matlab(stocks_path, index='tday', columns='stocks', 
                                data='cl')
    etf = pd.read_csv(etf_path, index_col='Date')

        
    #prepare the training and testing sets
    starts = '2007-01-01'
    ends = '2008-01-01'
    stks_train = stks[starts:ends]
    etf_train = etf[starts:ends] 
    stks_test = stks[ends:]
    etf_test = etf[ends:]
    
    
    ###########################################################################
    # Find the cointegrating portfolio
    ###########################################################################
    
    
    # johansen cointegration test for each stock vs SPY
    stks_list = list(stks.columns) 
    isCoint = pd.DataFrame(np.zeros([1,len(stks_list)], dtype=bool), 
                           columns=stks_list, index=['isCoint'])
    # set the confidence level
    confidence = 0.90
    if confidence==0.95: X=1
    elif confidence==0.99:x=2
    else: x=0
 
    for col in isCoint:
        # join the SPY to each of the stocks members in a Nx2 dataframe
        # clean for missing values
        y2 = etf_train.join(stks_train[col]).dropna()  
        # run johansen test for each of the N stocks vs SPY.
        if len(y2) > 250: 
            results = coint_johansen(y2, 0, 1, print_on_console=False)
            if results.lr1[0] > results.cvt[0,x]:
                isCoint[col] = True
 
 
    print('Johansen Test results for each stock vs SPY')
    print('--------------------------------------------------')
    print('Universe is {} stocks in the index'.format(len(stks_list)))
    print('Johansen Test over {} data points'.format(len(stks_train)))
    print('There are {} stocks that cointegrate with {:.2%} confidence'.
          format(np.count_nonzero(isCoint), confidence))           
    print('--------------------------------------------------')
    print()
    print()
   
    # capital allocation
    yN= repmat(isCoint, len(stks_train), 1) * stks_train
    yN = yN.replace(0, np.NaN)
    # the net market value of the long only portfolio 
    # We are using log price in this second test because we expect to rebalance 
    # this portfolio every day so that the capital on each stock is constant.
    log_mrk_val = pd.Series(np.sum(log(yN), axis=1), name='log_mrk_val')
    
    # Confirm that the portfolio cointegrates with SPY
    ytest = log(etf_train).join(log_mrk_val)
    print('Johansen Test results\nfor log value of portfolio vs log(SPY)')
    results = coint_johansen(ytest, 0, 1, print_on_console=True)
    
    if not (results.lr1[0] > results.cvt[0,0]):
        print("The portfolio doesn't cointegrate check it! stat:{:.4} < crit90%:{:.4}".
              format(results.lr1[0], results.cvt[0,0] ))
    else:    
        print("The portfolio cointegrate stat:{:.4} > crit90%:{:.4}, continue.".
              format(results.lr1[0], results.cvt[0,0] ))
       
        ###########################################################################
        # Apply linear mean-reversion model on test set
        ###########################################################################
        yNplus = (repmat(isCoint, len(stks_test), 1) * stks_test).join(etf_test)
        yNplus = yNplus.replace(0, np.NaN)
        
        
        weights = np.hstack((repmat(results.evec[1, 0], len(stks_test), len(isCoint.T)),
                             repmat(results.evec[0, 0], len(stks_test), 1)))
        
        
        #Log market value of long-short portfolio
        log_mrk_val = np.sum(log(yNplus) * weights, axis=1)
        
        # the parameter has lookahead bias, there should be some process to find it
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
        
        plt.show()

   
   
   
   
   
   
   
   
   
  

    
    
    
    