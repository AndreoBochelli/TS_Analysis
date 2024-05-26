import numpy as np
import pandas as pd
import tensorflow.data
from keras import initializers
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import crashes_deep

"""
more_time_steps = 4
cross_validation = False
period = 12 months
test = .1
MAPE = 8.6
"""


def LSTM_Bidirectional_1(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    EPOCHS = 10000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, activation='relu',
                           kernel_initializer=crashes_deep.uni, bias_initializer='zeros'),
                      input_shape=x_train.shape[-2:]),
        Bidirectional(LSTM(50, activation='tanh', kernel_initializer=crashes_deep.nor, bias_initializer='zeros')),
        # Dense(20),
        # Dropout(0.2),
        Dense(units=1),
    ])

    model.compile(optimizer=crashes_deep.adam, loss='mse')

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                batch_size = len(train_index) // 6
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                          steps_per_epoch=len(x_train[train_index]) // batch_size,
                          batch_size=batch_size, shuffle=True,
                          verbose=2,
                          callbacks=[
                              EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=2,
                                            mode='min'), ])
    else:
        batch_size = len(x_train) // 4
        model.fit(x_train, y_train, epochs=EPOCHS, shuffle=True,
                  verbose=1, steps_per_epoch=len(x_train) // batch_size,
                  callbacks=[
                      EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=2,
                                    mode='min'), ]
                  )

    score_train = model.evaluate(x_train, y_train, batch_size=10)
    score_test = model.evaluate(x_test, y_test, batch_size=10)
    print("in train MSE = ", score_train)
    print("in test MSE = ", score_test)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


"""
more_time_steps = 4
cross_validation = False
period = 12 months
test = .1
MAPE = 14.8
"""


def LSTM_Bidirectional_2(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    EPOCHS = 10000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, activation='relu',
                           kernel_initializer=crashes_deep.uni, bias_initializer=crashes_deep.nor),
                      input_shape=x_train.shape[-2:]),
        Bidirectional(
            LSTM(50, activation='tanh', kernel_initializer=crashes_deep.nor, bias_initializer=crashes_deep.uni)),
        # Dense(20),
        # Dropout(0.2),
        Dense(units=1),
    ])

    model.compile(optimizer=crashes_deep.sgd, loss='mse')

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                batch_size = len(train_index) // 6
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                          steps_per_epoch=len(x_train[train_index]) // batch_size,
                          batch_size=batch_size, shuffle=True,
                          verbose=2,
                          callbacks=[
                              EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1,
                                            mode='min'), ])
    else:
        batch_size = len(x_train) // 4
        model.fit(x_train, y_train, epochs=EPOCHS, shuffle=True,
                  verbose=1, steps_per_epoch=len(x_train) // batch_size,
                  callbacks=[
                      EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1,
                                    mode='min'), ]
                  )

    score_train = model.evaluate(x_train, y_train, batch_size=10)
    score_test = model.evaluate(x_test, y_test, batch_size=10)
    print("in train MSE = ", score_train)
    print("in test MSE = ", score_test)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


"""
more_time_steps = 4
cross_validation = False
period = 12 months
test = .1
MAPE = 14.6
"""


def LSTM_Bidirectional_3(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    EPOCHS = 10000
    CV_EPOCHS = 10
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, activation='relu',
                           kernel_initializer=crashes_deep.uni, bias_initializer=crashes_deep.nor),
                      input_shape=x_train.shape[-2:]),
        Bidirectional(
            LSTM(50, activation='tanh', kernel_initializer=crashes_deep.nor, bias_initializer=crashes_deep.uni)),
        # Dense(20),
        # Dropout(0.2),
        Dense(units=1),
    ])

    model.compile(optimizer=crashes_deep.sgd, loss='mse')

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for ep in range(CV_EPOCHS):
            print("CV_EPOCH # ", ep)
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                batch_size = len(train_index) // 6
                ds = tensorflow.data.Dataset.from_tensor_slices((x_train[train_index], y_train[train_index]))
                history = model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                                    steps_per_epoch=len(x_train[train_index]) // batch_size,
                                    batch_size=batch_size, shuffle=True,
                                    verbose=2,
                                    callbacks=[
                                        EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1,
                                                      mode='min'), ])
    else:
        batch_size = len(x_train) // 4
        model.fit(x_train, y_train, epochs=EPOCHS, shuffle=True,
                  verbose=1, steps_per_epoch=len(x_train) // batch_size,
                  callbacks=[
                      EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1,
                                    mode='min'), ]
                  )

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


"""
more_time_steps = 4
cross_validation = False
period = 10 months
test = .2
MAPE = 24.2
"""


def LSTM_Bidirectional_4(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    BATCH_SIZE = 10
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, activation="tanh"),
                      input_shape=x_train.shape[-2:]),
        Bidirectional(LSTM(50, return_sequences=True, activation="tanh")),
        Bidirectional(LSTM(10, return_sequences=True, activation=crashes_deep.softplus_layer)),
        Bidirectional(LSTM(5, activation=crashes_deep.softplus_layer)),
        Dense(units=1),
    ])
    sgd = SGD(learning_rate=0.001, momentum=0.95, nesterov=True)
    ad = Adam(learning_rate=0.006)
    model.compile(optimizer=crashes_deep.adam, loss='mse')
    EVALUATION_INTERVAL = 10
    EPOCHS = 15000

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for i in range(CV_EPOCHS):
            print('CV EPOCH', i)
            for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                BATCH_SIZE = len(train_index) // 10
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                          steps_per_epoch=len(x_train[train_index]) // BATCH_SIZE, batch_size=BATCH_SIZE,
                          verbose=1,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1,
                                                   mode='min'), ])
    else:
        model.fit(x_train, y_train, epochs=EPOCHS,
                  steps_per_epoch=EVALUATION_INTERVAL, batch_size=BATCH_SIZE,
                  verbose=1, callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1,
                                                      mode='min'), ])

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


"""
more_time_steps = 2
cross_validation = False
period = 10 months
test = .2
MAPE = 21.5
"""
"""
more_time_steps = 2
cross_validation = False
period = 6 months
test = .2
MAPE = 24.1
"""


def LSTM_Bidirectional_5(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    BATCH_SIZE = 10
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, activation="tanh", kernel_initializer=crashes_deep.uni,
                           bias_initializer=crashes_deep.uni),
                      input_shape=x_train.shape[-2:]),
        Bidirectional(
            LSTM(50, return_sequences=True, activation=crashes_deep.softplus_layer, kernel_initializer=crashes_deep.nor,
                 bias_initializer=crashes_deep.nor, )),
        Bidirectional(LSTM(10, return_sequences=False, activation="softsign", kernel_initializer=crashes_deep.nor,
                           bias_initializer=crashes_deep.uni, )),
        Dense(units=1, activation=crashes_deep.softplus_layer),
    ])
    sgd = SGD(learning_rate=0.001, momentum=0.95, nesterov=True)
    ad = Adam(learning_rate=0.002)
    model.compile(optimizer=ad, loss='mse')
    EVALUATION_INTERVAL = 10
    EPOCHS = 10000

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for i in range(CV_EPOCHS):
            print('CV EPOCH', i)
            for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                BATCH_SIZE = len(train_index) // 2
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                          steps_per_epoch=len(x_train[train_index]) // BATCH_SIZE, batch_size=BATCH_SIZE,
                          verbose=1,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1,
                                                   mode='min'), ])
        pred_batch_size = 2
    else:
        model.fit(x_train, y_train, epochs=EPOCHS,
                  steps_per_epoch=EVALUATION_INTERVAL, batch_size=BATCH_SIZE,
                  verbose=1, callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1,
                                                      mode='min'), ])
        pred_batch_size = 200

    pred_train = model.predict(x_train, batch_size=pred_batch_size)
    pred_test = model.predict(x_test, batch_size=pred_batch_size)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


"""
more_time_steps = 4
cross_validation = False
period = 10 months
test = .2
MAPE = 19.5
"""


def LSTM_Bidirectional_6(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    BATCH_SIZE = 2
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, activation="tanh", kernel_initializer=crashes_deep.uni,
                           bias_initializer=crashes_deep.uni),
                      input_shape=x_train.shape[-2:]),
        Bidirectional(
            LSTM(50, return_sequences=True, activation=crashes_deep.softplus_layer, kernel_initializer=crashes_deep.nor,
                 bias_initializer=crashes_deep.nor, )),
        Bidirectional(LSTM(10, return_sequences=False, activation="softsign", kernel_initializer=crashes_deep.nor,
                           bias_initializer=crashes_deep.uni, )),
        Dense(units=1, activation=crashes_deep.softplus_layer),
    ])
    sgd = SGD(learning_rate=0.001, momentum=0.95, nesterov=True)
    ad = Adam(learning_rate=0.002)
    model.compile(optimizer=ad, loss='mse')
    EVALUATION_INTERVAL = 10
    EPOCHS = 10000

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for i in range(CV_EPOCHS):
            print('CV EPOCH', i)
            for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                BATCH_SIZE = len(train_index) // 2
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                          steps_per_epoch=len(x_train[train_index]) // BATCH_SIZE, batch_size=BATCH_SIZE,
                          verbose=1,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1,
                                                   mode='min'), ])
    else:
        model.fit(x_train, y_train, epochs=EPOCHS,
                  steps_per_epoch=EVALUATION_INTERVAL, batch_size=BATCH_SIZE,
                  verbose=1, callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1,
                                                      mode='min'), ])
        pred_batch_size = 200

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_Bidirectional_7(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    EPOCHS = 5000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, activation='selu',
                           recurrent_activation='sigmoid',
                           kernel_initializer=crashes_deep.nor,
                           bias_initializer=crashes_deep.uni), input_shape=x_train.shape[-2:]),
        Bidirectional(LSTM(50, activation='tanh', kernel_initializer=crashes_deep.uni,
                           bias_initializer=crashes_deep.nor)),
        Dense(units=20),
        Dropout(0.2),
        Dense(units=1),
    ])

    model.compile(optimizer=crashes_deep.adam, loss='mse')

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                batch_size = len(train_index) // 6
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                                    steps_per_epoch=len(x_train[train_index]) // batch_size,
                                    batch_size=batch_size, shuffle=True,
                                    verbose=2,
                                    callbacks=[
                                        EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=2,
                                                      mode='min'), ])
    else:
        batch_size = 2
        model.fit(x_train, y_train, epochs=EPOCHS, shuffle=True,
                  verbose=1, steps_per_epoch=len(x_train) // batch_size,
                  callbacks=[
                      EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=2,
                                    mode='min'), ]
                  )

    score_train = model.evaluate(x_train, y_train)
    score_test = model.evaluate(x_test, y_test)
    print("in train MSE = ", score_train)
    print("in test MSE = ", score_test)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_Bidirectional_8(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x,
                         cross_validation: bool):
    EPOCHS = 1000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10
    init = initializers.RandomNormal(mean=0.0, stddev=1.0, seed=2024)
    init = initializers.RandomUniform(minval=0.0, maxval=1.0, seed=2024)
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, activation=crashes_deep.relu,
                           kernel_initializer=init, bias_initializer='zeros'), input_shape=x_train.shape[-2:]),
        Bidirectional(LSTM(50, activation='softsign', kernel_initializer=init, bias_initializer=init)),
        Dense(20),
        Dropout(0.2),
        Dense(units=1),
    ])

    model.compile(optimizer=crashes_deep.adam, loss='mse')

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                batch_size = len(train_index) // 2
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                          steps_per_epoch=len(x_train[train_index]) // batch_size,
                          batch_size=batch_size, shuffle=True,
                          verbose=2,
                          callbacks=[
                              EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=2,
                                            mode='min'), ])
    else:
        batch_size = len(x_train) // 10
        model.fit(x_train, y_train, epochs=EPOCHS, shuffle=True,
                  verbose=1, steps_per_epoch=len(x_train) // batch_size,
                  )

    score_train = model.evaluate(x_train, y_train, batch_size=10)
    score_test = model.evaluate(x_test, y_test, batch_size=10)
    print("in train MSE = ", score_train)
    print("in test MSE = ", score_test)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_Bidirectional_9(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    BATCH_SIZE = 10
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential([
        Bidirectional(LSTM(10, return_sequences=True, activation='tanh'), input_shape=x_train.shape[-2:]),
        Bidirectional(LSTM(5)),
        Dense(units=1),
    ])
    sgd = SGD(learning_rate=0.001, momentum=0.95, nesterov=True)
    ad = Adam(learning_rate=0.006)
    model.compile(optimizer=crashes_deep.adam, loss='mse')
    EVALUATION_INTERVAL = 10
    EPOCHS = 15000

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                BATCH_SIZE = len(train_index) // 2
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                          steps_per_epoch=len(x_train[train_index]) // BATCH_SIZE, batch_size=BATCH_SIZE,
                          verbose=1,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1,
                                                   mode='min'), ])
    else:
        model.fit(x_train, y_train, epochs=EPOCHS,
                  steps_per_epoch=EVALUATION_INTERVAL, batch_size=BATCH_SIZE,
                  verbose=1, callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1,
                                                      mode='min'), ])

    pred_train = model.predict(x_train, batch_size=BATCH_SIZE)
    pred_test = model.predict(x_test, batch_size=1)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_Bidirectional_10(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    BATCH_SIZE = 10
    EPOCHS = 150
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    init = initializers.RandomNormal(mean=0.0, stddev=1.0, seed=2024)
    # init = initializers.RandomUniform(minval=0.0, maxval=1.0, seed=2024)

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, kernel_initializer=crashes_deep.truncated_normal,
                           bias_initializer='zeros', activation=crashes_deep.leaky_elu),
                      input_shape=x_train.shape[-2:]),
        Bidirectional(LSTM(50, kernel_initializer=crashes_deep.truncated_normal,
                           bias_initializer='zeros', return_sequences=True, activation="tanh")),
        Bidirectional(LSTM(10, kernel_initializer=init,
                           bias_initializer='zeros', return_sequences=True, activation=crashes_deep.leaky_elu)),
        Bidirectional(LSTM(5, kernel_initializer=init,
                           bias_initializer='zeros', activation="tanh")),
        Dense(units=1),
    ])
    sgd = SGD(learning_rate=0.001, momentum=0.95, nesterov=True)
    ad = Adam(learning_rate=0.006)
    model.compile(optimizer=crashes_deep.nadam, loss='mse')
    EVALUATION_INTERVAL = 10

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for i in range(CV_EPOCHS):
            print('**************************   CV EPOCH  ******************************', i)
            for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                BATCH_SIZE = len(train_index) // 2
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                          steps_per_epoch=len(x_train[train_index]) // BATCH_SIZE, batch_size=BATCH_SIZE,
                          verbose=1)
    else:
        model.fit(x_train, y_train, epochs=EPOCHS,
                  steps_per_epoch=EVALUATION_INTERVAL, batch_size=BATCH_SIZE,
                  verbose=1, callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=1,
                                                      mode='min'), ])

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def LSTM_Bidirectional_11(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    BATCH_SIZE = 10
    EPOCHS = 1500
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    init = initializers.RandomNormal(mean=0.0, stddev=1.0, seed=2024)
    # init = initializers.RandomUniform(minval=0.0, maxval=1.0, seed=2024)

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, kernel_initializer=crashes_deep.truncated_normal,
                           bias_initializer='zeros', activation=crashes_deep.leaky_elu,
                           recurrent_activation="hard_sigmoid"),
                      input_shape=x_train.shape[-2:]),
        Bidirectional(LSTM(50, kernel_initializer=crashes_deep.truncated_normal,
                           bias_initializer='zeros', return_sequences=True, activation="tanh")),
        Bidirectional(LSTM(10, kernel_initializer=init,
                           bias_initializer='zeros', return_sequences=True, activation="hard_sigmoid")),
        Bidirectional(LSTM(5, kernel_initializer=init,
                           bias_initializer='zeros', activation="tanh")),
        Dense(units=1),
    ])
    sgd = SGD(learning_rate=0.001, momentum=0.95, nesterov=True)
    ad = Adam(learning_rate=0.006)
    model.compile(optimizer=crashes_deep.adam, loss='mse')
    EVALUATION_INTERVAL = 10

    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for i in range(CV_EPOCHS):
            print('**************************   CV EPOCH  ******************************', i)
            for _, (train_index, test_index) in enumerate(tscv.split(x_train)):
                BATCH_SIZE = len(train_index) // 2
                model.fit(x_train[train_index], y_train[train_index], epochs=EPOCHS_PER_SPLIT,
                          steps_per_epoch=len(x_train[train_index]) // BATCH_SIZE, batch_size=BATCH_SIZE,
                          verbose=1)
    else:
        model.fit(x_train, y_train, epochs=EPOCHS,
                  steps_per_epoch=EVALUATION_INTERVAL, batch_size=BATCH_SIZE,
                  verbose=1, callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=1000, verbose=1,
                                                      mode='min'), ])

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x
