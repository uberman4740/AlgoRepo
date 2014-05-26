

import pandas as pd
import scipy.io as sio
import numpy as np
from pandas import ExcelWriter
import matplotlib.pyplot as plt


def get_data_from_matlab(file_url, index, columns, data):
    """Description:*
    This function takes a Matlab file .mat and extract some 
    information to a pandas data frame. The structure of the mat
    file must be known, as the loadmat function used returns a 
    dictionary of arrays and they must be called by the key name
    
    Args:
        file_url: the ubication of the .mat file
        index: the key for the array of string date-like to be used as index
        for the dataframe
        columns: the key for the array of data to be used as columns in 
        the dataframe
        data: the key for the array to be used as data in the dataframe
    Returns:
        Pandas dataframe
         
    """
    
    import scipy.io as sio
    import datetime as dt
    # load mat file to dictionary
    mat = sio.loadmat(file_url)
    # define data to import, columns names and index
    cl = mat[data]
    stocks = mat[columns]
    dates = mat[index]
    
    # extract the ticket to be used as columns name in dataframe
    # to-do: list compression here
    columns = []
    for each_item in stocks:
        for inside_item in each_item:
            for ticket in inside_item:
                columns.append(ticket)
    # extract string ins date array and convert to datetimeindex
    # to-do list compression here
    df_dates =[]
    for each_item in dates:
        for inside_item in each_item:
            df_dates.append(inside_item)
    df_dates = pd.Series([pd.to_datetime(date, format= '%Y%m%d') for date in df_dates], name='date') 
    
    # construct the final dataframe
    data = pd.DataFrame(cl, columns=columns, index=df_dates)
    
    return data       



class strategy (object):
    def __init__(self, name):
        self.name = name
     
    def cl_std(self, cl, lookback=90):
        return pd.rolling_std(cl.diff(1) / cl.shift(1), window=lookback)
    
    def short(self, op, hi, cl, entryZscore=1, lookback=90, ma_window=20):
        up_gap_rtn = ((op - hi.shift(1)) /  hi.shift(1))
        tmp = (((op - hi.shift(1)) /  hi.shift(1)) < (self.cl_std(cl, lookback) * entryZscore)) 
        tmp2 = up_gap_rtn > 0
        tmp3 = cl < pd.rolling_mean(cl, window=ma_window )
        return up_gap_rtn * tmp * tmp2 * tmp3
        
    
    def long(self, op, lo, cl, entryZscore=1, lookback=90, ma_window=20):
        down_gap_rtn = ((op - lo.shift(1)) /  lo.shift(1)) 
        tmp =  ((op - lo.shift(1)) /  lo.shift(1)) > (self.cl_std(cl, lookback) * -entryZscore)
        tmp2 = down_gap_rtn < 0
        tmp3 = cl > pd.rolling_mean(cl, window=ma_window )
        return down_gap_rtn * tmp * tmp2 * tmp3

    def top_long_picks(self, df, topN=10):
        return df.rank(axis=1, ascending= True) <= topN
        
    def top_short_picks(self, df, topN=10):
        return df.rank(axis=1, ascending= False) <= topN
        
            
    def rtn_long(self, df, op, cl):
        pnl = np.sum(((cl-op) * df), axis=1) 
        mrk_val = np.sum((op * df), axis=1)
        rtn = np.cumsum(pnl / mrk_val, axis=0)
        rtn = rtn.fillna(method='pad')
        return rtn
        
    
    
    
    

if __name__ == "__main__":
    ################################################################
    # import data from MAT file
    ################################################################
    actual ='PC'
    
    if actual == 'PC':
        root_path = 'C:/Users/javgar119/Documents/Python/Data/'
    elif actual == 'MAC':
        root_path = '/Users/Javi/Documents/MarketData/'
    
    filename = 'example4_1.mat'   
    full_path = root_path + filename
    
    
    # get the data form mat file
    cl = get_data_from_matlab(full_path, index='tday', columns='stocks', data='cl')
    op = get_data_from_matlab(full_path, index='tday', columns='stocks', data='op')
    lo = get_data_from_matlab(full_path, index='tday', columns='stocks', data='lo')
    hi = get_data_from_matlab(full_path, index='tday', columns='stocks', data='hi')
    
    
    #export to csv files
    #cl.to_csv(root_path + 'cl.csv', sep=",")
    #op.to_csv(root_path + 'op.csv', sep=",")
    #lo.to_csv(root_path + 'lo.csv', sep=",")
    #hi.to_csv(root_path + 'hi.csv', sep=",")


    lookback = 90
    entryZscore= 1
    ma_window = 20
    topN= 10
    
    str = strategy('BuyPanic')
    longs = str.long(op=op, lo=lo, cl=cl, entryZscore=entryZscore, lookback=lookback,  ma_window=ma_window)
    top_long = str.top_long_picks(longs, topN)
    rtn = str.rtn_long(top_long, op=op, cl=cl)
    


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
    import datetime
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rtn)
    ax.set_title('SP500 DOWN GAP')
    ax.set_xlabel('Data points')
    ax.set_ylabel('cum rtn')
    ax.text(1200, 2, 'Sharpe: {:.4}'.format(sharpe))
    ax.text(1200, 1, 'APR: {:.4%}'.format(APR))
    plt.show()
    



