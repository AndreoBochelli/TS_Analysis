import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Bidirectional, Dropout, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from keras import initializers
import crashes_deep
import crashes_utils


def GRU_1(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    np.random.seed(2024)
    batch_size = 10
    EPOCHS = 150
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10
    init = initializers.RandomNormal(mean=0.0, stddev=1.0, seed=2024)
    # init = initializers.RandomUniform(minval=0.0, maxval=1.0, seed=2024)

    model = Sequential([
        GRU(100, kernel_initializer=init,
            bias_initializer='zeros', input_shape=x_train.shape[-2:]),
        Dense(20),
        Dropout(0.2),
        Dense(units=1),
    ])
    model.compile(optimizer=crashes_deep.adam, loss='mse')

    EVALUATION_INTERVAL = 10
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                                    shuffle=True, verbose=1, batch_size=len(train_index) // 10,
                                    callbacks=[
                                        EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1,
                                                      mode='min'), ])
    else:
                model.fit(x_train, y_train, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                            shuffle=True, verbose=1, batch_size=batch_size,
                            callbacks=[
                                EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1, mode='min'), ])

    pred_train = model.predict(x_train, batch_size=batch_size)
    pred_test = model.predict(x_test, batch_size=batch_size)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


"""
more_time_steps = 2
cross_validation = False
period = 6 months
test = .2
MAPE = 42.5
"""


def GRU_2(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    np.random.seed(2024)
    batch_size = 10
    EPOCHS = 10000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        GRU(100, return_sequences=True, activation=crashes_deep.softplus_layer,
            recurrent_activation=crashes_deep.softplus_layer,
            input_shape=x_train.shape[-2:]),
        GRU(5, activation=crashes_deep.softplus_layer, ),
        Dense(units=1),
    ])
    sgd = SGD(learning_rate=0.001, momentum=0.95, nesterov=True)
    model.compile(optimizer=crashes_deep.adam, loss='mse')

    EVALUATION_INTERVAL = 10

    if cross_validation:
        tscv = TimeSeriesSplit(n_splits=10, test_size=None, gap=0)
        for i in range(CV_EPOCHS):
            print('CV EPOCH', i)
            for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                history = model.fit(x_train, y_train, epochs=EPOCHS_PER_SPLIT, batch_size=len(train_index) // 2,
                                    shuffle=True, verbose=1,
                                    callbacks=[
                                        EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1,
                                                      mode='min'), ])
    else:
        model.fit(x_train, y_train, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                  shuffle=True, verbose=1, batch_size=2,
                  callbacks=[
                      EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def GRU_3(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    np.random.seed(2024)
    batch_size = 2
    EPOCHS = 15000
    CV_EPOCHS = 1000
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        GRU(100, return_sequences=True, activation=crashes_deep.elu,  kernel_initializer=crashes_deep.glorot_uniform,
            bias_initializer=crashes_deep.nor, input_shape=x_train.shape[-2:]),
        GRU(50, kernel_initializer=crashes_deep.glorot_uniform,  bias_initializer=crashes_deep.uni, activation='sigmoid'),
        Dense(units=1, activation='linear'),
    ])

    model.compile(optimizer=crashes_deep.adam, loss='mse')

    EVALUATION_INTERVAL = 10

    if cross_validation:
        tscv = TimeSeriesSplit(n_splits=10, test_size=None, gap=0)
        for i in range(CV_EPOCHS):
            print('CV EPOCH', i)
            for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                history = model.fit(x_train, y_train, epochs=EPOCHS_PER_SPLIT, batch_size=len(train_index) // 2,
                                    shuffle=True, verbose=1,
                                    callbacks=[
                                        EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1,
                                                      mode='min'), ])
    else:
        history = model.fit(x_train, y_train, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                            shuffle=True, verbose=1, batch_size=batch_size,
                            callbacks=[
                                EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1, mode='min'), ])

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x
