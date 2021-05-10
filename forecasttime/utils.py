import pandas as pd 
import numpy as np 

# Time Series dataframe transform to the format ready for machine learning and deep learning.
# Time related feature creation
# time series data standardization
# time series data decomposition

def series_to_supervised(data, n_in, n_out=1):
    """transform a time series dataframe to a supervised data format 
    
    parameters:
        data: time series data. If the original data is a dataframe, set the data as column values of the time series data
        n_in: the number of lagged terms to include to pull X
        n_out: the number of lead (future) terms to include to pull y
        
    return:
        numpy.ndarray: concat of X and y
        
    """
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values


# split a univariate sequence into samples
def split_sequence(sequence, n_steps, n_out):
    """split the time series sequence into array X of n_steps lagged terms and n_out lead (future) terms
    
    parameters:
        sequence: the time series data
        n_steps: the number of lagged terms to include to pull X
        n_out: the number of lead (future) terms to include to pull y
        
    Return:
        array_X: features array of n_steps lagged terms
        array_y: target array of n_out lead (future) term
    """
   
    X, y = list(), list()
    
    for i in range(len(sequence)-n_out-1):
        end_ix = i + n_steps
        end_iy = i + n_steps + n_out-1
        # check if we are beyond the sequence
        if (end_ix > len(sequence)-1) or (end_iy > len(sequence)-1): 
            break
            # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_iy]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# divide training and testing, default as 70:30
def train_test_split(dataset, test=0.3):
    """split the dataframe into train and test set
    
    Parameters:
        dataset: dataframe name
        test: 
            if test < 1: the ratio of the data for test
            if test >= 1: the number of observations for test 
    Return:
        two dataframes: one for train and one for test
               
    """
    
    if test < 1:
        test_size = int(len(dataset) * test)
        train_size = len(dataset) - test_size
        train, test = dataset[:train_size], dataset[train_size:]
        return train, test
    if test >= 1:
        test_size = test
        train_size = len(dataset) - test_size
        train, test = dataset[:train_size], dataset[train_size:]
        return train, test

