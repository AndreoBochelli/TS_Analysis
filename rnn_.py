from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from keras import initializers
import tensorflow.keras.optimizers as opts
from tensorflow.keras.callbacks import EarlyStopping
import crashes_deep


def RNN_1(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    batch_size = 10
    EPOCHS = 5000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential()
    model.add(SimpleRNN(units=100, activation="tanh", kernel_initializer=crashes_deep.nor,
                        bias_initializer=crashes_deep.uni, return_sequences=True, input_shape=x_train.shape[-2:]))
    # model.add(crashes_deep.softplus_layer)
    model.add(SimpleRNN(units=30, kernel_initializer=crashes_deep.nor,
                        bias_initializer=crashes_deep.uni, activation=crashes_deep.softplus_layer))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer=crashes_deep.adam)

    EVALUATION_INTERVAL = 3
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                model.fit(x_train[train_index], y_train[train_index], batch_size=len(train_index) // 10,
                          epochs=EPOCHS_PER_SPLIT, shuffle=True,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='min'), ])
        pred_batch_size = 2
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                  shuffle=True,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])
        pred_batch_size = 2

    pred_train = model.predict(x_train, batch_size=pred_batch_size)
    pred_test = model.predict(x_test, batch_size=pred_batch_size)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def RNN_2(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    init = initializers.RandomNormal(mean=0.0, stddev=1.0, seed=2024)
    # init = initializers.RandomUniform(minval=0.0, maxval=1.0, seed=2024)
    # lstm_model = Sequential()
    batch_size = 10
    EPOCHS = 100000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10
    model = Sequential()
    model.add(SimpleRNN(units=10, activation='tanh', kernel_initializer=init,
                        bias_initializer='zeros', input_shape=x_train.shape[-2:]))
    model.add(Dense(units=1))

    sgd = opts.SGD(learning_rate=0.00001, momentum=0.95, nesterov=True)
    adam = opts.Adam(learning_rate=0.006)
    model.compile(loss='mean_squared_error', optimizer=crashes_deep.adam)

    EVALUATION_INTERVAL = 13
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                model.fit(x_train[train_index], y_train[train_index], batch_size=len(train_index) // 10,
                          epochs=EPOCHS_PER_SPLIT,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='min'), ])
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])
    score_train = model.evaluate(x_train, y_train, batch_size=batch_size)
    score_test = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("in train MSE = ", score_train)
    print("in test MSE = ", score_test)

    pred_train = model.predict(x_train, batch_size=batch_size)
    pred_test = model.predict(x_test, batch_size=batch_size)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


"""
more_time_steps = 4
cross_validation = False
period = 12 months
test = .1
MAPE = 20.3
"""

def RNN_3(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    init = initializers.RandomNormal(mean=0.0, stddev=1.0, seed=2024)
    batch_size = 10
    EPOCHS = 5000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential()
    model.add(SimpleRNN(units=100, activation="tanh", kernel_initializer=crashes_deep.nor,
                        bias_initializer=crashes_deep.uni, return_sequences=True, input_shape=x_train.shape[-2:]))
    # model.add(crashes_deep.softplus_layer)
    model.add(SimpleRNN(units=30, kernel_initializer=crashes_deep.uni,
                        bias_initializer=crashes_deep.nor, activation="relu"))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer=crashes_deep.adam)

    EVALUATION_INTERVAL = 3
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                model.fit(x_train[train_index], y_train[train_index], batch_size=len(train_index) // 10,
                          epochs=EPOCHS_PER_SPLIT, shuffle=True,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='min'), ])
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                  shuffle=True,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])
    score_train = model.evaluate(x_train, y_train, batch_size=batch_size)
    score_test = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("in train MSE = ", score_train)
    print("in test MSE = ", score_test)

    pred_train = model.predict(x_train, batch_size=batch_size)
    pred_test = model.predict(x_test, batch_size=batch_size)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x

"""
more_time_steps = 4
cross_validation = True
period = 12 months
test = .1
MAPE = 19.8
"""
def RNN_4(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    init = initializers.RandomNormal(mean=0.0, stddev=1.0, seed=2024)
    batch_size = 10
    EPOCHS = 5000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential()
    model.add(SimpleRNN(units=100, activation="tanh", kernel_initializer=crashes_deep.nor,
                        bias_initializer=crashes_deep.uni, return_sequences=True, input_shape=x_train.shape[-2:]))
    # model.add(crashes_deep.softplus_layer)
    model.add(SimpleRNN(units=30, kernel_initializer=crashes_deep.uni,
                        bias_initializer=crashes_deep.nor, activation=crashes_deep.relu))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer=crashes_deep.adam)

    EVALUATION_INTERVAL = 3
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                model.fit(x_train[train_index], y_train[train_index], batch_size=len(train_index) // 10,
                          epochs=EPOCHS_PER_SPLIT, shuffle=True,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='min'), ])
        pred_batch_size = 2
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                  shuffle=True,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])
        pred_batch_size = 2

    pred_train = model.predict(x_train, batch_size= pred_batch_size)
    pred_test = model.predict(x_test, batch_size=pred_batch_size)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x

"""
more_time_steps = 4
cross_validation = False
period = 12 months
test = .1
MAPE = 18.2
"""
"""
more_time_steps = 4
cross_validation = True
period = 10 months
test = .2
MAPE = 31.2
"""
def RNN_4(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    batch_size = 10
    EPOCHS = 10000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential()
    model.add(SimpleRNN(units=100, activation=crashes_deep.relu, kernel_initializer=crashes_deep.glorot_uniform,
                        bias_initializer=crashes_deep.nor, return_sequences=True, input_shape=x_train.shape[-2:]))

    model.add(SimpleRNN(units=30, kernel_initializer=crashes_deep.glorot_uniform, activation=crashes_deep.softplus_layer,
                        bias_initializer=crashes_deep.glorot_uniform))
    model.add(Dense(units=1))

    sgd = opts.SGD(learning_rate=0.00001, momentum=0.95, nesterov=True)
    adam = opts.Adam(learning_rate=0.006)
    model.compile(loss='mean_squared_error', optimizer=opts.Adam(learning_rate=0.001, ))

    EVALUATION_INTERVAL = 13
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                model.fit(x_train[train_index], y_train[train_index], batch_size=len(train_index) // 10,
                          epochs=EPOCHS_PER_SPLIT,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='min'), ])
        pred_batch_size = 2
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])
        pred_batch_size = 2

    pred_train = model.predict(x_train, batch_size=pred_batch_size)
    pred_test = model.predict(x_test, batch_size=pred_batch_size)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x


def RNN_5(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    batch_size = 10
    EPOCHS = 10000
    CV_EPOCHS = 100
    EPOCHS_PER_SPLIT = 10

    model = Sequential()
    model.add(SimpleRNN(units=100, activation=crashes_deep.leaky_elu, kernel_initializer=crashes_deep.glorot_uniform,
                        bias_initializer='zeros', return_sequences=True, input_shape=x_train.shape[-2:]))

    model.add(SimpleRNN(units=30, kernel_initializer=crashes_deep.glorot_uniform, activation='softsign',
                        bias_initializer='zeros'))
    model.add(Dense(units=1, activation='sigmoid'))

    sgd = opts.SGD(learning_rate=0.00001, momentum=0.95, nesterov=True)
    adam = opts.Adam(learning_rate=0.006)
    model.compile(loss='mean_squared_error', optimizer=crashes_deep.rms_prop)

    EVALUATION_INTERVAL = 13
    if cross_validation:
        n_splits = 10
        test_size = None
        gap = 0
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        for _ in range(CV_EPOCHS):
            for i, (train_index, test_index) in enumerate(tscv.split(x_train)):
                model.fit(x_train[train_index], y_train[train_index], batch_size=len(train_index) // 10,
                          epochs=EPOCHS_PER_SPLIT,
                          callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='min'), ])
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0, patience=10000, verbose=1, mode='min'), ])

    pred_train = model.predict(x_train, batch_size=batch_size)
    pred_test = model.predict(x_test, batch_size=batch_size)

    return y, y_train, pred_train, pred_test, scaler_y, scaler_x
