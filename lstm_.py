import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.optimizers as opts
from keras import initializers
import crashes_deep


def LSTM_1(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    np.random.seed(2024)
    batch_size = 10
    EPOCHS = 10000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    lstm_model = Sequential([
        LSTM(50, activation="relu", return_sequences=True, input_shape=x_train.shape[-2:],
             kernel_initializer=crashes_deep.glorot_uniform, bias_initializer=crashes_deep.nor),
        LSTM(units=20, return_sequences=True, kernel_initializer=crashes_deep.nor,
             bias_initializer=crashes_deep.glorot_uniform, activation=crashes_deep.relu),
        Dropout(0.2),
        LSTM(units=10, activation=crashes_deep.softplus_layer, kernel_initializer=crashes_deep.glorot_uniform,
             bias_initializer=crashes_deep.nor),
        Dropout(0.2),
        Dense(units=1, ),
    ])

    lstm_model.compile(loss='mse', optimizer=opts.Adam(learning_rate=0.006, ))
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                batch_size = len(train_index) // 10
            lstm_model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT, batch_size=batch_size,
                           shuffle=True,
                           callbacks=[
                               EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])
    else:
        lstm_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_size, shuffle=True,
                       callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])

    pred_train = lstm_model.predict(x_train)
    pred_test = lstm_model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_2(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    import keras
    relu_layer = keras.layers.activation.ReLU(
        max_value=100,
        negative_slope=0.01,
        threshold=0,
    )
    batch_size = 10
    EPOCHS = 10000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    lstm_model = Sequential([
        LSTM(500, input_shape=x_train.shape[-2:], activation='tanh',
             kernel_initializer=crashes_deep.glorot_uniform, bias_initializer=crashes_deep.nor),
        Dense(units=1, activation='sigmoid', ),
    ])

    lstm_model.compile(loss='mse', optimizer=crashes_deep.adam)
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                batch_size_cv = len(train_index) // 10
                lstm_model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                               batch_size=batch_size_cv,
                               shuffle=True,
                               callbacks=[
                                   EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1, mode='min'), ])
    else:
        lstm_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_size, shuffle=True,
                       callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])

    pred_train = lstm_model.predict(x_train)
    pred_test = lstm_model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


"""
more_time_steps = 4
cross_validation = True
period = 12 months
test = .1
MAPE = 15.2
"""


def LSTM_3(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    import keras
    relu_layer = keras.layers.activation.ReLU(
        max_value=100,
        negative_slope=0.01,
        threshold=0,
    )
    batch_size = 10
    EPOCHS = 10000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    lstm_model = Sequential([
        LSTM(500, activation=crashes_deep.relu, input_shape=x_train.shape[-2:], kernel_initializer=crashes_deep.nor,
             bias_initializer=crashes_deep.nor),
        Dense(units=1, activation='sigmoid', ),
    ])

    lstm_model.compile(loss='mse', optimizer=crashes_deep.adam)
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                batch_size_cv = len(train_index) // 10
                lstm_model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                               batch_size=batch_size_cv,
                               shuffle=True,
                               callbacks=[
                                   EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1, mode='min'), ])
    else:
        lstm_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_size, shuffle=True,
                       callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])

    pred_train = lstm_model.predict(x_train)
    pred_test = lstm_model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x
