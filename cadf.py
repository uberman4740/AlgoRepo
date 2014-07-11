#!/usr/bin/python
# -*- coding: utf-8 -*-

# cadf.py

import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandas.io.data as web
import pprint
import statsmodels.tsa.stattools as ts

from pandas.stats.api import ols

# The first function, plot_price_series, takes a pandas DataFrame as input,
# with two columns given by the placeholder strings "ts1" and "ts2".
# These will be our pairs equities. The function simply plots the two price
# series on the same chart. This allows us to visually inspect whether any
# cointegration may be likely.
#
# We use the Matplotlib dates module to obtain the months from the datetime
# objects. Then we create a figure and a set of axes on which to apply the
# labeling/plotting. Finally, we plot the figure:

def plot_price_series(df, ts1, ts2):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('{} and %s Daily Prices'.format(ts1, ts2))
    plt.legend()
    plt.show()

# The second function, plot_scatter_series, plots a scatter plot of the two
# prices. This allows us to visually inspect whether a linear relationship
# exists between the two series and thus whether it is a good candidate for
# the OLS procedure and subsequent ADF test
def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()



# The third function, plot_residuals, is designed to plot the residual values
# from the fitted linear model of the two price series. This function requires
# that the pandas DataFrame has a "res" column, representing the residual
# prices:

def plot_residuals(df):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()

    plt.plot(df["res"])
    plt.show()

# Finally, the procedure is wrapped up in a __main__ function. The first task is
# to download the OHLCV data for both AREX and WLL from Yahoo Finance. Then we
# create a separate DataFrame, df, using the same index as the AREX frame to
# store both of the adjusted closing price values. We then plot the price series
# and the scatter plot.
#
# After the plots are complete the residuals are calculated by calling the
# pandas ols function on the WLL and AREX series. This allows us to calculate
# the β hedge ratio. The hedge ratio is then used to create a "res" column via
# the formation of the linear combination of both WLL and AREX.
#
# Finally the residuals are plotted and the ADF test is carried out on the
# calculated residuals. We then print the results of the ADF test:


if __name__ == "__main__":
    start = datetime.datetime(2013, 1, 1)
    end = datetime.datetime(2014, 1, 1)

    arex = web.DataReader("AREX", "yahoo", start, end)
    wll = web.DataReader("WLL", "yahoo", start, end)

    df = pd.DataFrame(index=arex.index)
    df["AREX"] = arex["Adj Close"]
    df["WLL"] = wll["Adj Close"]

    # Plot the two time series
    plot_price_series(df, "AREX", "WLL")

    # Display a scatter plot of the two time series
    plot_scatter_series(df, "AREX", "WLL")

    # Calculate optimal hedge ratio "beta"
    res = ols(y=df['WLL'], x=df["AREX"])
    beta_hr = res.beta.x

    # Calculate the residuals of the linear combination
    df["res"] = df["WLL"] - beta_hr * df["AREX"]

    # Plot the residuals
    plot_residuals(df)

    # Calculate and output the CADF test on the residuals
    cadf = ts.adfuller(df["res"])
    pprint.pprint(cadf)




