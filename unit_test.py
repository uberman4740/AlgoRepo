import unittest
import test

def test_gap(data, columns):
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(data, columns)
    serie = []
    for index, row in df.iterrows():
        picks = []
        tmp = (row * row.index.values)
        for item in tmp:
            if item != '':
                picks.append(item)
        serie.append(picks)
    return serie    
         
        
        
def eleva(x):
    return x**2
        
        
        
class SimplisticTest(unittest.TestCase):
    import pandas as pd
    data = [[True,False,False], [True,True,True], [False,False,True]]
    columns=('A','B','C')
    result = [['A','B'],['B'], ['B','C']]

    
    
    def testA(self):
        self.assertEqual(eleva(5), 25)

if __name__ == '__main__':
    unittest.main()