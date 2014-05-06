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
    
    data = Quandl.get(symbol, authtoken=authtoken, trim_start=start_date, trim_end=end_date, sort_order='asc')
    pd.DataFrame.to_csv(data, complete_path, header=header)
    # "self, path_or_buf, sep, na_rep, float_format, cols, header, index, index_label, mode, nanRep, encoding, quoting, line_terminator, chunksize, tupleize_cols, date_format)
    
    return 'Done'



if __name__ == "__main__":
    authtoken='ryDzS4euF3UoFtYwswQp'
    start_date = '1998-01-01'   #format "yyyy-mm-dd"
    end_date = '2014-01-01'     #format "yyyy-mm-dd"
    symbol =  ["GOOG/NYSE_EWC.4","GOOG/NYSE_EWA.4"]
    path_to_save ='/Users/Javi/Documents/MarketData/'
    filename = 'EWC_EWA_daily.csv'
    header = ['EWC','EWA']
   
    #'GOOG/NYSE_EWC'
    
    
    
    print(fromQuandl2csv(symbol,start_date=start_date, end_date=end_date, path=path_to_save, filename=filename, header=header))
    #plt.plot(data)
    #plt.show()
    