'''
Created on 14/11/2014

@author: javgar119
'''

# var.py

import datetime
import numpy as np
import pandas.io.data as web
from scipy.stats import norm
import pandas as pd

def var_cov_var(P, c, mu, sigma):
    """
    Variance-Covariance calculation of daily Value-at-Risk
    using confidence level c, with mean of returns mu
    and standard deviation of returns sigma, on a portfolio
    of value P.
    """
    alpha = norm.ppf(1 - c, mu, sigma)
    return P - P * (alpha + 1)




if __name__ == "__main__":
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2014, 1, 1)

    citi = web.DataReader("C", 'yahoo', start, end)
    apple = web.DataReader("AAPL", 'yahoo', start, end)
    portfolio = pd.DataFrame(dict(citi=citi["Adj Close"], apple=apple["Adj Close"]), index=citi.index)

    portfolio["citi ret"] = portfolio["citi"].pct_change()
    portfolio["apple ret"] = portfolio["apple"].pct_change()


    print(portfolio)
    P = 1e6  # 1,000,000 USD
    c = 0.95  # 99% confidence interval
    mu = [np.mean(portfolio["citi ret"]), np.mean(portfolio["apple ret"])]

    print('mu', mu)

    sigma = [np.std(portfolio["citi ret"]), np.std(portfolio["apple ret"])]
    print('sigma', sigma)

    var_co = np.correlate(portfolio["citi ret"], portfolio["apple ret"], 'full')
    print(var_co)


    # var = var_cov_var(P, c, mu, sigma)
    # print("Value-at-Risk: {0:.3f}".format(var))
