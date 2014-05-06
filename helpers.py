import pandas as pd
import Quandl 
import matplotlib.pyplot as plt



def fromQuandl2csv(symbol, start_date, end_date, path, filename, header):
    """
    Frequency:("daily"|weekly"|"monthly"|"quarterly"|"annual")
    Transformations:("diff"|"rdiff"|"normalize"|"cumul")
    sort_order=asc|desc
    Return last n rows: rows=n
    """
    authtoken='ryDzS4euF3UoFtYwswQp'
    complete_path = path + filename
    
    data = Quandl.get(symbol, authtoken=authtoken, trim_start=start_date, 
                      trim_end=end_date, sort_order='asc')
    pd.DataFrame.to_csv(data, complete_path, header=header)
   
    
    return 'Done'



if __name__ == "__main__":
    authtoken='ryDzS4euF3UoFtYwswQp'
    start_date = '2002-01-01'   #format "yyyy-mm-dd"
    end_date = '2014-01-01'     #format "yyyy-mm-dd"
    symbol =  ["GOOG/NYSE_EWC.4","GOOG/NYSE_EWA.4", 'GOOG/NYSE_IGE.4']
    path_to_save ='C:/Users/javgar119/Documents/Python/Data/'
    filename = 'EWC_EWA__IGE.csv'
    header = ['EWC','EWA','IGE']
   
    #'GOOG/NYSE_EWC'
    
    
    
    print(fromQuandl2csv(symbol,start_date=start_date, end_date=end_date, 
                         path=path_to_save, filename=filename, header=header))
    #plt.plot(data)
    #plt.show()
    