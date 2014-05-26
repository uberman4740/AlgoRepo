


def test_gap(df):
    
    import pandas as pd
    import numpy as np
    serie = []
    for index, row in df.iterrows():
        
        picks = []
        tmp = (row * row.index.values)
        for item in tmp:
            if item != '':
                picks.append(item)
        serie.append(picks)
    return serie    
        
        
        

    


input = pd.DataFrame({'A':[True,False,False],'B':[True,True,True],'C':[False,False,True]})
result = [['A','B'],['B'], ['B','C']]

