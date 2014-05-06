import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from functions import *



""" This is an implementation of the example 2.6  from Ernest Chan's
book ALGORITHMIC TRADING - Winning Strategies and Their Rationale
"""
 
     
if __name__ == "__main__":
   
    #import data from CSV file
    root_path = 'C:/Users/javgar119/Documents/Python/Data/'
    # the paths
    # MAC: '/Users/Javi/Documents/MarketData/'
    # WIN: 'C:/Users/javgar119/Documents/Python/Data'
    filename = 'EWC_EWA__IGE.csv'
    full_path = root_path + filename
    full_dataframe = pd.read_csv(full_path, index_col='Date')
    # create a series with the data range asked
    start_date = '2006-01-20'
    end_date = '2012-06-11'
    data =  subset_dataframe(full_dataframe, start_date, end_date)

    x = data['EWC']
    y = data['EWA']
    z = data['IGE']
    k = polyfit(x,y,1)
    xx = linspace(min(x),max(x),1000)
    yy = polyval(k,xx)
 
    lookback = 60
    modelo2 = pd.ols(y=y, x=x, window_type='rolling', window=lookback)
    data = data[lookback-1:]
    betas = modelo2.beta
    #calculate the number of units for the strategy
    numunits = subtract(data['EWC'], multiply(betas['x'], 
                                              data['EWA']))

    model = sm.OLS(y, x)
    results = model.fit()
    print(results.params)
    
    
    # cointegration test
    resultsCOIN = cointegration_test(x,y)
    print('****** COINTEGRATION TEST RESULTS *****')
    print('cointegration t-stat: {}'.format(round(resultsCOIN[0],4)))
    print('p-value: {}'.format(round(round(resultsCOIN[1],4))))
    print('usedlag: {}'.format(round(round(resultsCOIN[2],4))))
    print('nobs: {}'.format(round(resultsCOIN[3],4)))
    print('critical values: {}'.format(resultsCOIN[4]))
        
    

    #*************************************************
    # plotting the charts
    #*************************************************    
    #plot of numunits
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(numunits)
    ax.set_title('EWC-hedgeRatio*EWA')
    ax.set_xlabel('Data points')
    ax.set_ylabel('Numunits')
    ax.text(1100, 25, 'the number of units ')
    

    #plot of datapoints
    #fig1 = plt.figure()
    ax = fig.add_subplot(212)
    ax.plot(x,y,'o')
    ax.plot(xx,yy,'r')
    ax.set_title('EWC vs. EWA')
    ax.set_xlabel('EWC')
    ax.set_ylabel('EWA')
    
    plt.show()
    
    
    
