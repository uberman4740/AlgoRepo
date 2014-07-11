#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on Oct 24, 2014

@author: Javier Garcia, javier.macro.trader@gmail.com
'''
import sqlite3
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt

#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 14/08/2014

@author: Javier Garcia, javier.macro.trader@gmail.com
'''

import os
import pandas as pd
import numpy as np

from data import DataHandler
from event import MarketEvent

from pprint import pprint
# pylint: disable=too-many-instance-attributes
# Eight is reasonable in this case.
class HistoricSQLiteDataHandler(DataHandler):
    """
    HistoricSQLiteDataHandler is designed to read a SQLite database for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.

    ARG:
        database_path: the system path where the SQLite database is stored.
                
        symbol_list: list containing the name of the symbols
                to read in the database.
                The SQL consult is expected to have the following structure:
                [date-time, open, high, low, close, volume, adjusted_close]

        IMPORTANT: for different symbols in differents time-zones you must assure
                    the data is correctly syncronized. Athena does not check
                    this.

    """
    def __init__(self, events, database, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the database and a list of symbols.

        It will be assumed that all price data is in a table called 
        'symbols', where the field 'symbol' is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the database
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.database = database
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self.bar_index = 0
        self.all_data_dic = {}  # access data in list form for testing

        self._open_convert_database_data()
    
    def _connect_to_database(self, database, flavor='sqlite3'):
        """
        Connect to the database ....
        :param database: full path to SQLite3 database to connect
        """
        if flavor == 'sqlite3':
            try:
                connection = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
                return connection
            except sqlite3.Error as err:
                print('Error connecting database', err.args[0])
        # TODO: this leg is not finished
        elif flavor == 'SQLAlchemy':
            try:
                engine = sqlalchemy.create_engine('sqlite://'+database)
                return engine
            except sqlalchemy.exc as err:
                print('Error connecting database', err)
        
    def _get_prices(self, conn, symbol, cols):
        """
        Query the database and returns a dataframe with the chosen 7 columns.
        :param conn:
        :param symbol:
        :param cols:
        """
        values_qry = '''SELECT {},{},{},{},{},{},{}
                        FROM prices WHERE symbol="{}"'''.format(cols[0],
                                                                cols[1],
                                                                cols[2],
                                                                cols[3],
                                                                cols[4],
                                                                cols[5],
                                                                cols[6],
                                                                symbol)
        return pd.read_sql(values_qry, conn, index_col='price_date')

        

    def _open_convert_database_data(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.

        For this handler it will be assumed that the data is
        taken from DTN IQFeed. Thus its format will be respected.
        """
        comb_index = None

        columns = ['price_date',
                   'open_price',
                   'high_price',
                   'low_price',
                   'close_price',
                   'volume',
                   'adjusted_price']
        connection = self._connect_to_database(self.database)


        for symbol in self.symbol_list:
            self.symbol_data[symbol] = self._get_prices(connection, symbol, columns)

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[symbol].index
            else:
                comb_index.union(self.symbol_data[symbol].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[symbol] = []

        # Reindex the dataframes
        for symbol in self.symbol_list:
            self.all_data_dic[symbol] = self.symbol_data[symbol].\
                reindex(index=comb_index, method=None)

            self.symbol_data[symbol] = self.symbol_data[symbol].\
                reindex(index=comb_index, method=None).iterrows()


    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for symbol_gen in self.symbol_data[symbol]:
            yield symbol_gen

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            raise KeyError("Symbol is not available in the data set.")
        else:
            if not bars_list:
                raise KeyError('latest_symbol_data has not been initialized.')
            else:
                return bars_list[-1]

    def get_latest_bars(self, symbol, bars=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            raise KeyError("Symbol is not available in the data set.")
        else:
            if not bars_list:
                raise KeyError('latest_symbol_data has not been initialized.')
            else:
                return bars_list[-bars:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            raise KeyError("Symbol is not available in the data set.")
        else:
            if not bars_list:
                raise KeyError ('latest_symbol_data has not been initialized.')
            else:
                return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            raise KeyError("Symbol is not available in the data set.")
        else:
            if not bars_list:
                raise KeyError ('latest_symbol_data has not been initialized.')
            else:
                return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol, val_type, bars=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(symbol, bars)
        except KeyError:
            raise KeyError("Symbol is not available in the data set.")
        else:
            if not bars_list:
                raise KeyError ('latest_symbol_data has not been initialized.')
            else:
                return np.array([getattr(b[1], val_type) for b in bars_list])

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for symbol in self.symbol_list:
            try:
                bars = self._get_new_bar(symbol).__next__()
            except StopIteration:
                self.continue_backtest = False
            else:
                if bars is not None:
                    self.latest_symbol_data[symbol].append(bars)
        self.events.put(MarketEvent())


#
#
#
# import queue
# if __name__ == '__main__':
#     from_location = 'PC'
#     if from_location == 'PC':
#         DATABASE = 'C:/Users/javgar119/Documents/Dropbox/SEC_MASTER/securities_master.SQLITE'
#     elif from_location == 'MAC':
#         DATABASE = '//Users/Javi/Dropbox/MarketDB/securities_master.SQLITE'
#
#     LIST_OF_SYMBOLS = ['SPY_QL', 'EWA_QL']
#     EVENT = queue.Queue()
#     STORE = HistoricSQLiteDataHandler(EVENT, DATABASE, LIST_OF_SYMBOLS)
#
#     print(type(STORE))
#
#
#
#     for dummy in range(5483):
#         STORE.update_bars()
#         # DATETIME1 = STORE.get_latest_bar_datetime('SPY_QL')
#         # SPY = STORE.get_latest_bar_value('SPY_QL', val_type='close_price')
#         bar_value = STORE.get_latest_bar('SPY_QL')
#         # DATETIME2 = STORE.get_latest_bar_datetime('EWA_QL')
#         # EWA = STORE.get_latest_bar_value('EWA_QL', val_type='close_price')
#         print(bar_value)
#         # print(DATETIME1, SPY, '  -  ', DATETIME2, EWA)
#
#





#
#
# def _get_prices(conn, symbols, cols):
#     """
#
#     :param conn:
#     :param symbol:
#     :param cols:
#     """
#     symbol_data = dict()
#     for each_symbol in symbols:
#         values_qry = '''SELECT {},{},{},{},{},{},{}
#                         FROM prices WHERE symbol="{}"'''.format(cols[0],
#                                                                 cols[1],
#                                                                 cols[2],
#                                                                 cols[3],
#                                                                 cols[4],
#                                                                 cols[5],
#                                                                 cols[6],
#                                                                 each_symbol)
#         symbol_data[each_symbol] = pd.read_sql(values_qry,
#                                                conn,
#                                                index_col='price_date')
#     return symbol_data
#
#
# def _connect_to_database(database, flavor='sqlite3'):
#     """
#     Connect to the database ....
#     :param database: full path to SQLite3 database to connect
#     """
#     if flavor == 'sqlite3':
#         try:
#             connection = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
#             return connection
#         except sqlite3.Error as err:
#             print('Error connecting database', err.args[0])
#     # TODO: this leg is not finished
#     elif flavor == 'SQLAlchemy':
#         try:
#             engine = sqlalchemy.create_engine('sqlite://' + database)
#             return engine
#         except sqlalchemy.exc as err:
#             print('Error connecting database', err)
#
#
#
#
#
# if __name__ == '__main__':
#     from_location = 'PC'
#     if from_location == 'PC':
#         database = 'C:/Users/javgar119/Documents/Dropbox/SEC_MASTER/securities_master.SQLITE'
#     elif from_location == 'MAC':
#         database = '//Users/Javi/Dropbox/MarketDB/securities_master.SQLITE'
#
#     conn = _connect_to_database(database)
#
#     symbols = ['SP500_QL', 'USO_QL']
#     columns = ['price_date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'adjusted_price']
#     symbol_data = _get_prices(conn, symbols, columns)
