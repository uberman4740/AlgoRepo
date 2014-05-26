

import pandas as pd
import scipy.io as sio
import numpy as np

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
    
def gap_finder(op, lo, cl, hi, lookback=90, entryZscore=1, is_down_gap=True, topN=10):
    """Descrption:
    This function calculates where a stocks has gaped agains T-1
    highs or low.
    It takes the time series of open, high, close and low prices and compute
    the distance from T+0 open to T-1 high (low) and compares agains the
    historical up (down) gaps over a lookback period.
    Distances grater that one (1) standard deviation are clasified as gaps. 
    UpGap: high(T-1) - open(T+0) > entryZscore(UpGaps in lookback period)  
    DownGap = open(T-0) - low(T+1) > entryZscore(DownGaps in lookback period)
    Args:
        op: dataframe containing open prices for a universe of stocks. size(NxM)
        lo: dataframe containing low prices for a universe of stocks. size(NxM)
        hi: dataframe containing high prices for a universe of stocks. size(NxM)
        cl: dataframe containing close prices for a universe of stocks. size(NxM)
        index and columns in these dataframes must be aligned
        lookback: periods to be used in calculation od zscore
        entryZscore: thereshold to determinate if theres a gap
    return:
        Dataframe: size (NxM), index and columns as original dataframes.
        +1 = UpGap
        -1 = DownGap
        0 NaN 
        
    """
    
    cl_std = pd.rolling_std(cl.diff(1) / cl.shift(1), window=lookback)
    up_gap_rtn= ((op - hi.shift(1)) /  hi.shift(1))
    down_gap_rtn = ((op - lo.shift(1)) /  lo.shift(1))
    
    if is_down_gap == False:
        # UP GAPS = 1
        # open is higher that last high
        AA = up_gap_rtn > 0 
        # gao is lower that z entry score
        BB = up_gap_rtn < (cl_std*entryZscore)
        gap= AA * BB * down_gap_rtn
        gap = (gap.rank(axis=1, ascending= False)) <= topN
        return gap
        
    elif is_down_gap == True:
        # DOWN GAPS = -1
        # open is lower that last low
        HH = down_gap_rtn < 0
        # gao is lower that z entry score
        MM = down_gap_rtn > (-cl_std*entryZscore)
        gap = HH * MM * down_gap_rtn 
        gap = gap.rank(axis=1, ascending= True) <= topN
        return gap
    





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

    ################################################################
    # strategy 
    ################################################################
    lookback = 90
    entryZscore= 1
    topN= 10
    ma_window = 20
     
    down_gap = gap_finder(op=op, lo=lo, cl=cl, hi=hi, lookback=lookback, 
                          entryZscore=entryZscore, is_down_gap=True, topN=topN)
    up_gap = gap_finder(op=op, lo=lo, cl=cl, hi=hi, lookback=lookback, 
                        entryZscore=entryZscore, is_down_gap=False)
    under_ma = cl < pd.rolling_mean(cl, window=ma_window)
    over_ma = cl > pd.rolling_mean(cl, window=ma_window)
    
    
    numunits_long = over_ma * down_gap
    
    
    # show the pick for each day
    for index, row in numunits_long.iterrows():
        picks = []
        tmp = (row * row.index.values)
        for item in tmp:
            if item != '':
                picks.append(item)
        print(picks)
        
    
    # pnl and return calculations
    
    pnl = np.sum((cl-op) * numunits_long, axis=1)
    mrk_val = np.sum(op*numunits_long, axis=1)
    rtn = pnl / mrk_val

    acum_rtn = pd.DataFrame(np.cumsum(rtn))
    acum_rtn = acum_rtn.fillna(method='pad')
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
    ax.plot(acum_rtn)
    ax.set_title('SP500 DOWN GAP')
    ax.set_xlabel('Data points')
    ax.set_ylabel('cum rtn')
    ax.text(1200, 2, 'Sharpe: {:.4}'.format(sharpe))
    ax.text(1200, 1, 'APR: {:.4%}'.format(APR))
    plt.show()



