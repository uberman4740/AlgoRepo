
from datetime import *
import datetime
from numpy import *
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import scipy.io as sio
import pandas as pd

 
def normcdf(X):
    (a1,a2,a3,a4,a5) = (0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
    L = abs(X)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / sqrt(2*pi)*exp(-L*L/2.) * (a1*K + a2*K*K + a3*pow(K,3) + a4*pow(K,4) + a5*pow(K,5))
    if X < 0:
        w = 1.0-w
    return w

def vratio(a, lag = 2, cor = 'hom'):
    """ the implementation found in the blog Leinenbock  
    http://www.leinenbock.com/variance-ratio-test/
    """
    #t = (std((a[lag:]) - (a[1:-lag+1])))**2;
    #b = (std((a[2:]) - (a[1:-1]) ))**2;
 
    n = len(a)
    mu  = sum(a[1:n]-a[:n-1])/n;
    m=(n-lag+1)*(1-lag/n);
    #print( mu, m, lag)
    b=sum(square(a[1:n]-a[:n-1]-mu))/(n-1)
    t=sum(square(a[lag:n]-a[:n-lag]-lag*mu))/m
    vratio = t/(lag*b);
 
    la = float(lag)
     
    if cor == 'hom':
        varvrt=2*(2*la-1)*(la-1)/(3*la*n)
 
    elif cor == 'het':
        varvrt=0;
        sum2=sum(square(a[1:n]-a[:n-1]-mu));
        for j in range(lag-1):
            sum1a=square(a[j+1:n]-a[j:n-1]-mu);
            sum1b=square(a[1:n-j]-a[0:n-j-1]-mu)
            sum1=dot(sum1a,sum1b);
            delta=sum1/(sum2**2);
            varvrt=varvrt+((2*(la-j)/la)**2)*delta
 
    zscore = (vratio - 1) / sqrt(float(varvrt))
    pval = normcdf(zscore);
 
    return  vratio, zscore, pval
 
def hurst2(ts):
    """ the implementation found in the blog Leinenbock  
    http://www.leinenbock.com/calculation-of-the-hurst-exponent-to-test-for-trend-and-mean-reversion/
    """
    tau = []; lagvec = []
    #  Step through the different lags
    for lag in range(2,100):
        #  produce price difference with lag
        pp = subtract(ts[lag:],ts[:-lag])
        #  Write the different lags into a vector
        lagvec.append(lag)
        #  Calculate the variance of the differnce vector
        tau.append(sqrt(std(pp)))
 
    #  linear fit to double-log graph (gives power)
    m = polyfit(log10(lagvec),log10(tau),1)
    # calculate hurst
    hurst = m[0]*2.0
    # plot lag vs variance
    #plt.plot(lagvec,tau,'o')
    #plt.show()
 
    return hurst

def hurst(ts):
    """ the implewmentation on the blog http://www.quantstart.com
    http://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing
    Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

def half_life(ts):  
    """ this function calculate the half life of mean reversion
    """
    # calculate the delta for each observation. 
    # delta = p(t) - p(t-1)
    delta_ts = diff(ts)
        # calculate the vector of lagged prices. lag = 1
    # stack up a vector of ones and transpose
    lag_ts = vstack([ts[1:], ones(len(ts[1:]))]).T
   
    # calculate the slope (beta) of the deltas vs the lagged values 
    beta = linalg.lstsq(lag_ts, delta_ts)
    
    # compute half life
    half_life = log(2) / beta[0]
    
    return half_life[0]

def random_walk(seed=1000, mu = 0.0, sigma = 1, length=1000):
    """ this function creates a series of independent, identically distributed values
    with the form of a random walk. Where the best prediction of the next value is the present
    value plus some random variable with mean and variance finite 
    We distinguish two types of random walks: (1) random walk without drift (i.e., no constant
    or intercept term) and (2) random walk with drift (i.e., a constant term is present).  
    The random walk model is an example of what is known in the literature as a unit root process.
    RWM without drift: Yt = Yt−1 + ut
    RWM with drift: Yt = δ + Yt−1 + ut
    """
    
    ts = []
    for i in range(length):
        if i == 0:
            ts.append(seed)
        else:    
            ts.append(mu + ts[i-1] + random.gauss(0, sigma))

    return ts

def subset_dataframe(data, start_date, end_date):
    start = data.index.searchsorted(start_date)
    end = data.index.searchsorted(end_date)
    return data.ix[start:end]

def cointegration_test(y, x):
    ols_result = sm.OLS(y, x).fit()
    return ts.adfuller(ols_result.resid, maxlag=1)


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


def my_path(loc):
    if loc == 'PC':
        root_path = 'C:/Users/javgar119/Documents/Python/Data/'
    elif loc == 'MAC':
        root_path = '/Users/Javi/Documents/MarketData/'
    return root_path