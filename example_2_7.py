from johansen_test import coint_johansen

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from functions import *


#from numpy import *
#from numpy.linalg import *


if __name__ == "__main__":
   
    #import data from CSV file
    root_path = 'C:/Users/javgar119/Documents/Python/Data/'
    # the paths
    # MAC: '/Users/Javi/Documents/MarketData/'
    # WIN: 'C:/Users/javgar119/Documents/Python/Data'
    filename_x = 'EWC_EWA_IGE_daily.csv'
    #filename_y = 'ECOPETROL_ADR.csv'
    full_path_x = root_path + filename_x
    #full_path_y = root_path + filename_y
    data = pd.read_csv(full_path_x, index_col='Date')
    #create a series with the data range asked
    #start_date = '2006-01-03'
    #end_date = '2012-04-17'
    #data =  subset_dataframe(data, start_date, end_date)
    
    results = coint_johansen(data, 0, 1)
    w = results.evec
    
    print(type(w))
    print(w[:,0])
        
    
    # Calculate portfolio
    # with the result eigenvector uses as the weigth, proced to
    # calculate the portfolio.
    
    
    
    