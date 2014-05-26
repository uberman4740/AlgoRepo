import unittest
import pandas as pd
import numpy as np
import datetime
from pandas.util.testing import  (assert_series_equal, assert_almost_equal,
                                 assert_frame_equal)
from example_4_1_copy import strategy



class gap_finder_tests(unittest.TestCase):
    loockbak = 10
    
    
    todays_date = datetime.datetime.now().date()
    index = pd.date_range(todays_date-datetime.timedelta(1), periods=20, freq='D')
    columns = ['a', 'b']
    op=[100.00, 99.90, 102.00, 105.00, 104.00]
    hi=[101.50, 101.40, 102.00, 105.00, 107.00]
    lo=[99.00 
 99.90 
 99.00 
 102.00 
 104.00 
]
    
         
    def simple_df(self):
        self.assertEqual(1, 1, 'is equal')

    def test_cl_std(self):
        index = list(range(20))
        data = list(range(0,20))
        columns=['a']
        expected =  [0, 0, 0, 0, 0, 0,
                     0.324380, 0.133125, 0.075713, 0.049584, 0.035222,
                     0.026399, 0.020563, 0.016488, 0.013526, 0.011302,
                     0.009589, 0.008239, 0.007158, 0.006277] 
        
        df_expected = pd.DataFrame(expected, index=index, columns=columns)   
        df = pd.DataFrame(data, index=index, columns=columns)
        str = strategy('BuyPanic')
        result = str.cl_std(cl=df, lookback=5)
        result = result.fillna(0)
        result = np.round(result,6)
        
        return assert_frame_equal(result, df_expected)

    def test_up_gap_rtn(self):
        todays_date = datetime.datetime.now().date()
        index = pd.date_range(todays_date-datetime.timedelta(1), periods=20, freq='D')
        columns = ['a', 'b']
        op=[100.00, 99.90, 102.00, 105.00, 104.00]
        hi=[101.50, 101.40, 102.00, 105.00, 107.00] 
        
        
        
        
        





if __name__ == '__main__':
    main()
