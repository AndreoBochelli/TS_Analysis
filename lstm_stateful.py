import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adagrad

import crashes_deep


def LSTM_online_1(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    n_batch = 1
    EPOCHS = 150
    CV_EPOCHS = 1
    np.random.seed(2024)
    init = crashes_deep.nor
    init = crashes_deep.uni

    model = Sequential([
        LSTM(15, activation="relu", recurrent_activation="hard_sigmoid", return_sequences=True, kernel_initializer=init,
             bias_initializer=init,
             batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2]), stateful=True),
        LSTM(units=10, activation="tanh", return_sequences=True, kernel_initializer=init, bias_initializer=init,
             recurrent_activation="hard_sigmoid",
             stateful=True),
        # Dropout(0.2),
        LSTM(units=5, return_sequences=True, kernel_initializer=init, recurrent_activation="hard_sigmoid",
             bias_initializer=init, stateful=True),
        # Dropout(0.2),
        Dense(units=1),
    ])

    model.compile(loss="mean_squared_error", optimizer=crashes_deep.adam)

    for i in range(EPOCHS):
        print("EPOCH #", i)
        if cross_validation:
            n_splits = 10
            test_size = None
            gap = 0
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
            for _ in range(CV_EPOCHS):
                for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                    model.fit(x_train[train_index], y_train[train_index], epochs=1,
                              batch_size=1, verbose=1, shuffle=False,
                              callbacks=[
                                  EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=1, mode='min'), ])
                    model.reset_states()
        else:
            model.fit(x_train, y_train, batch_size=n_batch, epochs=1, verbose=1, shuffle=False,
                      callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=1, mode='min'), ])
            model.reset_states()

    # online forecast
    pred_train = np.zeros(y_train.shape)
    print('TRAINING:\n')
    for i in range(len(x_train)):
        trainX, trainy = x_train[i], y_train[i]
        trainX = trainX.reshape(1, x_train.shape[1], x_train.shape[2])
        yhat = model.predict(trainX, batch_size=1)
        print('>Expected=%f, Predicted=%f' % (trainy, yhat))
        pred_train[i] = yhat

    pred_test = np.zeros(y_test.shape)
    print('VALIDATION:\n')
    for i in range(len(x_test)):
        testX, testy = x_test[i], y_test[i]
        testX = trainX.reshape(1, x_test.shape[1], x_test.shape[2])
        yhat = model.predict(testX, batch_size=1)
        print('>Expected=%f, Predicted=%f' % (testy, yhat))
        pred_test[i] = yhat

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_online_2(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    n_batch = 1
    EPOCHS = 10
    CV_EPOCHS = 1
    np.random.seed(2024)
    init = crashes_deep.nor
    init = crashes_deep.uni
    model = Sequential([
        LSTM(50, activation="tanh", return_sequences=True, recurrent_activation="hard_sigmoid",
             stateful=True, batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2])),
        LSTM(units=100, stateful=True, return_sequences=False),
        Dense(units=1),
    ])

    model.compile(loss="mean_squared_error", optimizer=crashes_deep.adam)

    for i in range(EPOCHS):
        print("EPOCH #", i)
        if cross_validation:
            n_splits = 10
            test_size = None
            gap = 0
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
            for _ in range(CV_EPOCHS):
                for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1,
                              shuffle=False)
                    model.reset_states()
        else:
            model.fit(x_train, y_train, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
            model.reset_states()

    # online forecast
    pred_train = np.zeros(y_train.shape)
    print('TRAINING:\n')
    for i in range(len(x_train)):
        trainX, trainy = x_train[i], y_train[i]
        trainX = trainX.reshape(1, x_train.shape[1], x_train.shape[2])
        yhat = model.predict(trainX, batch_size=1)
        print('>Expected=%f, Predicted=%f' % (trainy, yhat))
        pred_train[i] = yhat

    pred_test = np.zeros(y_test.shape)
    print('VALIDATION:\n')
    for i in range(len(x_test)):
        testX, testy = x_test[i], y_test[i]
        testX = trainX.reshape(1, x_test.shape[1], x_test.shape[2])
        yhat = model.predict(testX, batch_size=1)
        print('>Expected=%f, Predicted=%f' % (testy, yhat))
        pred_test[i] = yhat

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_online_3(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    n_batch = 1
    EPOCHS = 1000
    CV_EPOCHS = 1
    np.random.seed(2024)
    init = crashes_deep.nor
    init = crashes_deep.uni

    model = Sequential([
        LSTM(50, activation="tanh", kernel_initializer=crashes_deep.nor,
             bias_initializer=crashes_deep.uni, return_sequences=True),
        LSTM(units=100, kernel_initializer=crashes_deep.nor,
             bias_initializer=crashes_deep.uni, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, activation="tanh", return_sequences=False),
        Dropout(0.2),
        Dense(units=1),
    ])

    model.compile(loss="mean_squared_error", optimizer=crashes_deep.adam)

    for i in range(EPOCHS):
        print("EPOCH #", i)
        if cross_validation:
            n_splits = 10
            test_size = None
            gap = 0
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
            for _ in range(CV_EPOCHS):
                for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                    model.fit(x_train[train_index], y_train[train_index], epochs=1,
                              callbacks=[
                                  EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=1, mode='min'), ],
                              batch_size=1, verbose=2, shuffle=False)
                    model.reset_states()
        else:
            model.fit(x_train, y_train, epochs=1, batch_size=n_batch, verbose=1, shuffle=False,
                      callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=1, mode='min'), ])
            model.reset_states()

    # online forecast
    pred_train = np.zeros(y_train.shape)
    print('TRAINING:\n')
    for i in range(len(x_train)):
        trainX, trainy = x_train[i], y_train[i]
        trainX = trainX.reshape(1, x_train.shape[1], x_train.shape[2])
        yhat = model.predict(trainX, batch_size=1)
        print('>Expected=%f, Predicted=%f' % (trainy, yhat))
        pred_train[i] = yhat

    pred_test = np.zeros(y_test.shape)
    print('VALIDATION:\n')
    for i in range(len(x_test)):
        testX, testy = x_test[i], y_test[i]
        testX = trainX.reshape(1, x_test.shape[1], x_test.shape[2])
        yhat = model.predict(testX, batch_size=1)
        print('>Expected=%f, Predicted=%f' % (testy, yhat))
        pred_test[i] = yhat

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_batch(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    x = np.concatenate((x_train, x_test), axis=0)
    learn_end = (len(x_train) + len(x_test)) // 2
    end = 2 * learn_end
    y = y[:end]
    x_train = x[: learn_end]
    x_test = x[learn_end: end]
    y_train = y[: learn_end]
    seed = 2024
    np.random.seed(seed)
    n_epoch = 10000
    n_batch = x_train.shape[0]
    init = crashes_deep.nor
    init = crashes_deep.uni

    model = Sequential()
    model.add(LSTM(units=100, activation='relu', return_sequences=True,
                   batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2]), stateful=True,
                   kernel_initializer=init, bias_initializer='zeros'))
    model.add(LSTM(units=100, batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2]),
                   activation='sigmoid', stateful=True,
                   kernel_initializer=init, bias_initializer='zeros'))
    model.add(Dense(1))

    model = Sequential()
    model.add(LSTM(100, batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2]), return_sequences=True,
                   stateful=True, activation="softsign", kernel_initializer=crashes_deep.nor,
                   bias_initializer='zeros'))

    model.add(LSTM(50, activation="softsign", stateful=True, return_sequences=True, kernel_initializer=crashes_deep.nor,
                   bias_initializer='zeros'))
    model.add(LSTM(10, activation="softsign", stateful=True, kernel_initializer=crashes_deep.nor,
                   bias_initializer='zeros'))
    model.add(Dense(units=1, activation='linear', ))

    model.compile(loss='mean_squared_error', optimizer=crashes_deep.adam)

    # fit network
    for i in range(n_epoch):
        print('Iteration', i)
        if cross_validation:
            raise Exception("No cross validation for such a model")
        else:
            model.fit(x_train, y_train, epochs=1, batch_size=n_batch, verbose=1, shuffle=False,
                      callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=1, mode='min'), ])
            model.reset_states()

    # batch forecast
    pred_train = model.predict(x_train, batch_size=n_batch)
    pred_test = model.predict(x_test, batch_size=n_batch)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_copy_weights_1(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    x = np.concatenate((x_train, x_test), axis=0)
    learn_end = 81
    end = len(x)
    x_train = x[:learn_end]
    x_test = x[learn_end:end]
    y_train = y[:learn_end]
    y_test = y[learn_end:end]
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    n_batch = 9
    assert not len(x_train) % n_batch
    n_epoch = 700

    # design network
    model = model_copy_weights_3(n_batch, x_train)
    sgd = SGD(learning_rate=0.0004, momentum=0.95, nesterov=True)
    adam = Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer=adam)

    # fit network
    for i in range(n_epoch):
        if cross_validation:
            raise Exception("No cross validation for such a model")
        else:
            model.fit(x_train, y_train, epochs=1, batch_size=n_batch, verbose=1, shuffle=False,
                      callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1, mode='min'), ])
            model.reset_states()

    # re-define the batch size
    n_batch = 1
    # re-define model
    new_model = model_copy_weights_3(n_batch, x_train)
    # copy weights
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)
    # compile model
    new_model.compile(loss='mean_squared_error', optimizer=adam)

    print('TRAINING:\n')
    pred_train = new_model.predict(x_train, batch_size=1)

    pred_test = np.zeros(y_test.shape)
    print('VALIDATION:\n')
    for i in range(len(x_test)):
        testX, testy = x_test[i], y_test[i]
        testX = testX.reshape(1, x_test.shape[1], x_test.shape[2])
        yhat = new_model.predict(testX, batch_size=n_batch)
        print('>Expected=%f, Predicted=%f' % (testy, yhat))
        pred_test[i] = yhat

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_copy_weights_2(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    # configure network
    x = np.concatenate((x_train, x_test), axis=0)
    learn_end = (len(x_train) + len(x_test)) // 2
    end = 2 * learn_end
    y = y[:end]
    x_train = x[: learn_end]
    x_test = x[learn_end: end]
    y_train = y[: learn_end]
    y_test = y[learn_end: end]
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    n_batch = x_train.shape[0] // 2
    n_epoch = 1000
    assert not len(x_train) % n_batch

    # design network
    model = model_copy_weights_3(n_batch, x_train)

    sgd = SGD(learning_rate=0.0004, momentum=0.95, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='sgd')
    # fit network
    for i in range(n_epoch):
        if cross_validation:
            raise Exception("No cross validation for such a model")
        else:
            model.fit(x_train, y_train, epochs=1, batch_size=n_batch, verbose=1, shuffle=False,
                      callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1, mode='min'), ])
            model.reset_states()
    # re-define the batch size
    n_batch = 1
    # re-define model
    new_model = model_copy_weights_3(n_batch, x_train)
    # copy weights
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)
    # compile model
    new_model.compile(loss='mean_squared_error', optimizer=sgd)

    print('TRAINING:\n')
    pred_train = new_model.predict(x_train, batch_size=1)

    pred_test = np.zeros(y_test.shape)
    print('VALIDATION:\n')
    for i in range(len(x_test)):
        testX, testy = x_test[i], y_test[i]
        testX = testX.reshape(1, x_test.shape[1], x_test.shape[2])
        yhat = new_model.predict(testX, batch_size=n_batch)
        print('>Expected=%f, Predicted=%f' % (testy, yhat))
        pred_test[i] = yhat

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def model_copy_weights_3(n_batch, x_train):
    model = Sequential([
        LSTM(50, stateful=True, activation="tanh", batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2]),
             return_sequences=True,
             kernel_initializer=crashes_deep.nor, bias_initializer=crashes_deep.nor),
        LSTM(units=100, stateful=True, return_sequences=False, kernel_initializer=crashes_deep.nor,
             bias_initializer=crashes_deep.nor),
        Dense(units=1),
    ])
    return model


def model_copy_weights_4(n_batch, x_train):
    model = Sequential()
    model.add(LSTM(100, batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2]), return_sequences=True,
                   stateful=True, activation="relu", ))
    model.add(LSTM(7, activation="sigmoid", stateful=True, return_sequences=True, ))
    model.add(LSTM(5, activation="sigmoid", stateful=True, ))
    model.add(Dense(units=1, ))
    return model
