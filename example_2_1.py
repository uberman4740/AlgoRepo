
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from functions import *

""" This is an implementation of the examples 2.1 to 2.5 from Ernest chan's
book ALGORITHMIC TRADING - Winning Strategies and Their Rationale
"""
     
if __name__ == "__main__":

    #import data from CSV file
    root_path = '/Users/Javi/Documents/MarketData/'
    filename = 'USDCAD_daily.csv'
    full_path = root_path + filename
    full_dataframe = pd.read_csv(full_path, index_col='Date')
    
    # create a series with the data range asked
    start_date = '23/07/2007'
    end_date = '28/03/2012'
    data =  subset_dataframe(full_dataframe, start_date, end_date)
           
            
    
    # COMPUTE STATISTICS
        
    # compute the ADF test with the statsmodels pack
    adf = ts.adfuller(data['Rate'], maxlag=1, regression="c")
    print('****** ADF TEST RESULTS *****')
    print('ADF t-stat: {}'.format(round(adf[0],4)))
    print('p-value: {}'.format(round(adf[1],4)))
    print('usedlag: {}'.format(round(adf[2],4)))
    print('nobs: {}'.format(adf[3]))
    print('critical values: {}'.format(adf[4]))
    # computes the hurst exponent
    print('****** HURST EXPONENT ******')
    print('Hurst: {}'.format(round(hurst(data['Rate']),4)))
    # compute the variance ratio test
    a = log(array(data['Rate']))
    vrtest = vratio(a, cor = 'het', lag = 20)
    print('****** VARIANCE RATIO TEST ******')
    print ('stat: {}, zscore: {}, p-value: {}'.format(round(vrtest[0],4), round(vrtest[1],4), round(vrtest[2],4)))
    # compute half life of the time serie
    print('****** HALF LIFE ******')
    hl = int(half_life(data['Rate']))
    print('half life: {}'.format(hl))
        
    # LINEAR STRATEGY
    # A linear trading strategy means that the number of units or shares of a
    # unit portfolio we own is proportional to the negative z-score of the 
    # price series of that portfolio.
    
    
    lookback = int(hl)
    moving_mean = pd.rolling_mean(data, window=lookback) 
    moving_std = pd.rolling_std(data,window=lookback)
    
    print(len(moving_mean))
    
    print(len(moving_std))
    print(len(data))
    
    z_score = divide(subtract(data['Rate'] , moving_mean['Rate']), moving_std['Rate'])
    numunits = multiply(z_score, -1)
    pnl = divide(multiply(numunits[:-1], diff(data['Rate'])),-numunits[:-1])
    
    # plot accumulative % PnL
        # some especifications for the charts
    font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 16,
        }
    
    
    plt.plot(cumsum(pnl))
    plt.title('Acumulative result z-score strategy', fontdict=font)
    plt.show()
 
