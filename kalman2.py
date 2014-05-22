
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from numpy.matlib import repmat
    import matplotlib.pyplot as plt
    
    


    from functions import *
    from numpy.matlib import repmat
    import numpy as np


       #import data from CSV file
    root_path = '/Users/Javi/Documents/MarketData/'
    # the paths
    # MAC: '/Users/Javi/Documents/MarketData/'
    # WIN: 'C:/Users/javgar119/Documents/Python/Data/'
    filename = 'EWC_EWA_daily2.csv'   #version 2 is bloomberg data
    full_path = root_path + filename
    data = pd.read_csv(full_path, index_col='Date')
    print(data.head(10))
    x_ticket = 'EWA'
    y_ticket = 'EWC'
    
    y = np.asarray(data[y_ticket])
    x = pd.DataFrame(data[x_ticket])
    index = x.index

    
    # Augment x with ones to  accomodate possible offset in the regression
    # between y vs x
    x['ones'] = np.ones(len(x))
    x = np.asarray(x)
    

    # delta=1 gives fastest change in beta, delta=0.000....
    # 1 allows no change (like traditional linear regression).
    delta=0.0001
    
    
    yhat=np.zeros(shape=(len(x),1))     # measurement prediction
    e=np.zeros(shape=(len(x),1))        # measurement prediction error
    Q=np.zeros(shape=(len(x),1))        # measurement prediction error variance

  
  
    # For clarity, we denote R(t|t) by P(t).
    # initialize R, P and beta.
    R = np.zeros((2,2));
    P = np.zeros((2,2));
    beta = np.zeros((2, len(x)))
    Vw = delta/(1-delta)*np.eye(2)
    Ve=0.001
    

    # Given initial beta and R (and P)


    for t in range(len(y)):
        if t > 0:
            beta[:, t] = beta[:, t-1]    # state prediction. Equation 3.7
            R= P + Vw                      # state covariance prediction. Equation 3.8
        
            yhat[t] = x[t, :].dot(beta[:, t])                             # measurement prediction. Equation 3.9
            Q[t] = x[t, :].dot(R).dot(x[t, :].T) + Ve         # measurement variance prediction. Equation 3.10
           
            # Observe y(t)
            e[t] = y[t] - yhat[t]                       # measurement prediction error
            K = R.dot(x[t, :].T) / Q[t]     # Kalman gain
            
            beta[:, t] = beta[:, t] + K * e[t]              # State update. Equation 3.11
            
            P= R - K * x[t, :] * R              # State covariance update. Euqation 3.12
    sqrt_Q = np.sqrt(Q)
    
    beta = pd.DataFrame(beta.T, index= index, columns=('beta', 'x'))
    e = pd.DataFrame(e, index=index)
 

    y2 = pd.DataFrame([x[:, 0],y]).transpose()
    y2 = y2.set_index(index)

    #plt.plot(beta['x'])
    #plt.show
    
    # trade signal 
    long_entry = e < -sqrt_Q   # a long position means we should buy EWC
    long_exit = e > -sqrt_Q
    short_entry = e > sqrt_Q
    short_exit =  e < sqrt_Q
    #print(short_entry, short_exit)
    """
    OJO ACA PORQUE ESTA MACHETIADO
    REVISAR LAS POSICIONES
    CUANDO LARGO ES COMPRAR EWC DICE CHANG
    """
    numunits_long= np.zeros((len(y2),1))
    numunits_long = pd.DataFrame(np.where(long_entry,-1,0))
    
    numunits_short= np.zeros((len(y2),1))
    numunits_short = pd.DataFrame(np.where(short_entry,1,0))

    numunits = numunits_long + numunits_short
   
    


    #compute the $ position for each asset
    AA = pd.DataFrame(repmat(numunits,1,2))
    #print(AA.head(10))
    BB = pd.DataFrame(-beta['beta'])
    BB['ones'] = np.ones((len(beta)))
    #print(BB.head(10))
    position = multiply(multiply(AA, BB), y2)
    #print(position.head(10))
    #print (BB.head(50))

    # compute the daily pnl in $$
    
    pnl = sum(multiply(position[:-1], divide(diff(data,axis = 0), y2[:-1])),1)
    
        
    # gross market value of portfolio
    mrk_val = pd.DataFrame.sum(abs(position), axis=1)
    mrk_val = mrk_val[:-1]
   
    # return is P&L divided by gross market value of portfolio
    rtn = pnl / mrk_val

    acum_rtn = pd.DataFrame(cumsum(rtn))
    acum_rtn = acum_rtn.fillna(method='pad')

    # compute performance statistics
    sharpe = (np.sqrt(252)*np.mean(rtn)) / np.std(rtn)
    APR = np.prod(1+rtn)**(252/len(rtn))-1
    ##################################################
    # print the results
    ##################################################
    print('Price spread Sharpe: {:.4}'.format(sharpe))
    print('Price Spread APR: {:.4%}'.format(APR))
    


#*************************************************
    # plotting the chart
    #*************************************************    
    #plot of numunits
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax1.plot(acum_rtn)
    
    #ax2 = fig.add_subplot(212)
    #ax2.plot(e)
    
    plt.show()






