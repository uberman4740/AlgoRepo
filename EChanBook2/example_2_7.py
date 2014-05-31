from johansen_test import coint_johansen
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from numpy.matlib import repmat

# from numpy import *
# from numpy.linalg import *


if __name__ == "__main__":

    # import data from CSV file
    root_path = 'C:/Users/javgar119/Documents/Python/Data/'
    # the paths
    # MAC: '/Users/Javi/Documents/MarketData/'
    # WIN: 'C:/Users/javgar119/Documents/Python/Data'
    filename = 'GLD_SLV_daily.csv'
    full_path = root_path + filename
    data = pd.read_csv(full_path, index_col='Date')
    # create a series with the data range asked
    # start_date = '2010-01-13'
    # end_date = '2014-05-13'
    # data = subset_dataframe(data, start_date, end_date)
    # print('data import is {} lines'.format(str(len(data))))
    # print(data.head(10))
    # print(data.tail(5))


    # johansen test with non-zero offset but zero drift, and with the lag k=1.
    results = coint_johansen(data, 0, 1)

    # those are the weigths of the portfolio
    # the first eigenvector because it shows the strongest cointegration relationship

    w = results.evec[:, 0]

    print('Best eigenvector is: {}.'.format(str(w)))

    # (net) market value of portfolio
    # this is the syntetic asset we are going to trade. A freshly
    # new mean reverting serie compose of the three assets in
    # proportions given by the eigenvector
    yport = pd.DataFrame.sum(w * data, axis=1)
    print(yport)
    # print(yport.tail(10))


    print('Hurst: {}'.format(round(hurst(yport), 4)))


    # LINEAR STRATEGY
    # A linear trading strategy means that the number of units or shares of a
    # unit portfolio we own is proportional to the negative z-score of the
    # price series of that portfolio.
    HalfLf = half_life(yport)
    print('Half Life is: {}'.format(str(HalfLf)))
    lookback = int(HalfLf)



    moving_mean = pd.rolling_mean(yport, window=lookback)
    moving_std = pd.rolling_std(yport, window=lookback)
    # print(moving_mean)
    z_score = (yport - moving_mean) / moving_std
    numunits = pd.DataFrame(z_score * -1, columns=['numunits'])
    # print(numunits.tail(10))

    AA = repmat(numunits, 1, 2)
    BB = multiply(repmat(w, len(data), 1), data)
    position = pd.DataFrame(multiply(AA, BB))
    # print(position.tail(10))

    pnl = sum(divide(multiply(position[:-1], diff(data, axis=0)), data[:-1]), 1)
    # print(pnl)
    # gross market value of portfolio
    mrk_val = pd.DataFrame.sum(abs(position), axis=1)
    # return is P&L divided by gross market value of portfolio
    rtn = cumsum(pd.DataFrame(pnl / mrk_val, columns=['rtn']))
    print(rtn.iloc[-2])

    plt.plot(rtn[25:])
    plt.show()
