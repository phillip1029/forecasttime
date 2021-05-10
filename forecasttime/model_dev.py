from forecasttime.utils import series_to_supervised, train_test_split

# fit a model
def model_fit(name, train, config):
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

# forecast with a pre-fit model
def model_predict(name, model, history, config):
    if name == "mlp":
        # unpack config
        n_input, _, _, _ = config
        # prepare data
        x_input = array(history[-n_input:]).reshape(1, n_input)
        # forecast
        yhat = model.predict(x_input, verbose=0)
        return yhat[0]