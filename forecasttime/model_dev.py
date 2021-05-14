import os

from forecasttime.utils import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.layers import GRU, Embedding, LSTM, TimeDistributed, ConvLSTM2D

import numpy as np


def difference(data, interval):
    """ difference dataset
    
    parameters:
        data: dataset to be differenced
        interval: the interval between the two elements to be differenced. 
        
    return: 
        dataset: with the length = len(data) - interval
        
    """
    return [data[i] - data[i - interval] for i in range(interval, len(data))]

# fit a model
def model_fit(name, train, config): 
    """ build and fit different deep learning models per config settings
    
    parameters:
        name: 
            "mlp": multilayer perceptron model (MLP)
            "cnn": convolutional neural network model (cnn)
            "lstm": Long short-term memory network model (lstm)
            "cnn-lstm": hybrid model of cnn and lstm
            "conv-lstm": hybrid model of convolutional neural network model 
            
        config:
            "mlp": [n_input, n_nodes, n_epochs, n_batch]
            "cnn": [n_input, n_filters, n_kernel, n_epochs, n_batch]
            "lstm": n_input, n_nodes, n_epochs, n_batch, n_diff
            "cnn-lstm":[n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch]
            "conv-lstm": [n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch]
        train:
            training data
            
    return: 
        model: which can be saved for future usage.
    
    """
    from forecasttime.utils import series_to_supervised
    
    if name == "mlp":
        # unpack config
        n_input, n_nodes, n_epochs, n_batch = config
        # prepare data
        data = series_to_supervised(train, n_input)
        train_x, train_y = data[:, :-1], data[:, -1]

        # define MLP model
        model = Sequential()
        model.add(Dense(n_nodes, activation='relu', input_dim=n_input))

        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        # model fit
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
        return model

    if name == "cnn":
        # unpack config
        n_input, n_filters, n_kernel, n_epochs, n_batch = config
        # prepare data
        data = series_to_supervised(train, n_input)
        train_x, train_y = data[:, :-1], data[:, -1]
        train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
        # define model
        model = Sequential()
        model.add(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(n_input, 1)))
        model.add(Conv1D(n_filters, n_kernel, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        # fit
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
        return model
    if name == "lstm":
        # unpack config
        n_input, n_nodes, n_epochs, n_batch, n_diff = config
        # prepare data
        if n_diff > 0:
            train = difference(train, n_diff)
        data = series_to_supervised(train, n_input)
        train_x, train_y = data[:, :-1], data[:, -1]
        train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
        # define model
        model = Sequential()
        model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, 1)))
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        # fit
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
        return model
    if name == "cnn-lstm":
        # unpack config
        n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
        n_input = n_seq * n_steps
        # prepare data                                       
        data = series_to_supervised(train, n_input)
        train_x, train_y = data[:, :-1], data[:, -1]
        train_x = train_x.reshape((train_x.shape[0], n_seq, n_steps, 1))
        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(None,n_steps,1))))
        model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D()))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(n_nodes, activation='relu'))
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        # fit
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
        return model

    if name == "conv-lstm":
        # unpack config
        n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
        n_input = n_seq * n_steps
        # prepare data                         
        data = series_to_supervised(train, n_input)
        train_x, train_y = data[:, :-1], data[:, -1]
        train_x = train_x.reshape((train_x.shape[0], n_seq, 1, n_steps, 1))
        # define model
        model = Sequential()
        model.add(ConvLSTM2D(n_filters, (1,n_kernel), activation='relu', input_shape=(n_seq, 1, n_steps, 1)))
        model.add(Flatten())
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        # fit
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
        return model

# forecast with a pre-fit model
def model_predict(name, model, history, config):
    if name == "mlp":
        # unpack config
        n_input, _, _, _ = config
        # prepare data
        x_input = np.array(history[-n_input:]).reshape(1, n_input)
        # forecast
        yhat = model.predict(x_input, verbose=0)
        return yhat[0]
    if name == "cnn":
        # unpack config
        n_input, _, _, _, _ = config
        # prepare data
        x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))
        # forecast
        yhat = model.predict(x_input, verbose=0)
        return yhat[0]
    if name == "lstm":
        # unpack config
        n_input, _, _, _, n_diff = config
        # prepare data
        correction = 0.0
        if n_diff > 0:
            correction = history[-n_diff]
            history = difference(history, n_diff)
        x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))
        # forecast
        yhat = model.predict(x_input, verbose=0)
        return correction + yhat[0]
    if name == "cnn-lstm":
        # unpack config
        n_seq, n_steps, _, _, _, _, _ = config
        n_input = n_seq * n_steps
        # prepare data                                    
        x_input = np.array(history[-n_input:]).reshape((1, n_seq, n_steps, 1))
        # forecast
        yhat = model.predict(x_input, verbose=0)
        return yhat[0]
    if name == "conv-lstm":
        # unpack config
        n_seq, n_steps, _, _, _, _, _ = config
        n_input = n_seq * n_steps
        # prepare data                 
        x_input = np.array(history[-n_input:]).reshape((1, n_seq, 1, n_steps, 1))
        # forecast
        yhat = model.predict(x_input, verbose=0)
        return yhat[0]
    
    