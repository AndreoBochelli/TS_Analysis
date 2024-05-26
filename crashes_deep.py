import keras.optimizers as kopts
import keras.activations as lacts
import keras.layers.activation as acts
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.optimizers as opts
from keras import initializers as inits
from sklearn import preprocessing
import crashes_utils
import kernel_methods


def crashes_for_networks(df: pd.DataFrame, arcsined: bool):
    x = np.arange(0, len(crashes_utils.crashes))
    y = df['Number of crashes']
    scaler_x = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler_y = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    if not arcsined:
        x = scaler_x.fit_transform(x)
        y = scaler_y.fit_transform(y)

    df['Number of crashes'] = y
    end = len(x)
    learn_end = int(end * crashes_utils.train_portion)
    x_train = x[: learn_end, ]
    x_test = x[learn_end: end, ]
    y_train = y[: learn_end]
    y_test = y[learn_end:end]
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    return y, x, y_train, x_train, y_test, x_test, scaler_y, scaler_x


def do_predict(do_specific_predict: object, cross_validation: bool, arcsined=False, full_pic=False, more_time_steps=5):
    """
      :param arcsined: specifies whether to apply arcsin modification to the data
      :param full_pic: specifies whether to show forecast only or together with the model fitted
      :param cross_validation: whether cross validation is applied or not
      :param do_specific_predict: specifies which kind of network to apply
      :param more_time_steps: no of extra time steps (beyond the single one) to predict next time step)
      :return: nothing
      """
    if arcsined:
        emergencies = arcsining_data(crashes_utils.crashes)
    else:
        emergencies = crashes_utils.crashes.copy()
        crashes_utils.crashes['Ticks'] = pd.to_datetime(crashes_utils.crashes['Ticks']).dt.date
        crashes_utils.crashes = crashes_utils.crashes.set_index('Ticks')
        emergencies.index = crashes_utils.crashes.index

    crashes_utils.crashes = emergencies

    y, x, y_train, x_train, y_test, x_test, scaler_y, scaler_x = crashes_for_networks(emergencies, arcsined)
    if do_specific_predict == kernel_methods.kernels:
        more_time_steps = 0
    y, y_train, x_train, x_test = crashes_utils.multi_steps(x_test, x_train, y, y_train, more_time_steps)

    y, pred_train, pred_test = output_pic(full_pic, arcsined, pd.to_datetime(emergencies['Ticks']).dt.date[more_time_steps:],
                                             do_specific_predict(y, y_train, x_train, y_test, x_test,
                                                                 scaler_y, scaler_x, cross_validation))

    if arcsined:
        y = pd.DataFrame(y)
        y.columns = ['Number of crashes']
        crashes_utils.Trigo.restore(y, 0, len(y))
        pred_train = pd.DataFrame(pred_train)
        pred_train.columns = ['Number of crashes']
        crashes_utils.Trigo.restore(pred_train, 0, len(pred_train))
        pred_test = pd.DataFrame(pred_test)
        pred_test.columns = ['Number of crashes']
        crashes_utils.Trigo.restore(pred_test, 0, len(pred_test))
        pred = pd.DataFrame(np.concatenate((pred_train, pred_test), axis=0))
        pred.columns = ['Number of crashes']
        if not full_pic:
            crashes_utils.prediction_plot(pd.DataFrame(y), pd.DataFrame(y_train), pd.DataFrame(pred_test))
        else:
            import seaborn as sns
            sns.set_style("whitegrid")
            crashes_utils.seaborn_plot(y.to_numpy(), y_train, pd.to_datetime(emergencies['Ticks']).dt.date[more_time_steps:], pred.to_numpy())


def arcsining_data(crashes: pd.DataFrame):
    crashes['Ticks'] = pd.to_datetime(crashes['Ticks']).dt.date
    crashes = crashes.set_index('Ticks')
    crashes_ = crashes.copy()
    crashes.index = np.arange(0, len(crashes))
    crashes_utils.Trigo.scale = crashes['Number of crashes'].max()
    crashes['Number of crashes'] = crashes['Number of crashes'] / crashes['Number of crashes'].max()
    arg_max = pd.Series(crashes['Number of crashes'].index[crashes['Number of crashes']
                                                           == crashes['Number of crashes'].max()]).mean()
    crashes_utils.Trigo.index_max = int(arg_max)
    crashes.apply(crashes_utils.Trigo.trigo, axis=1)
    crashes.index = crashes_.index
    crashes['Ticks'] = crashes.index
    return crashes


def output_pic(full_pic: bool, arcsined: bool, dates: pd.Series, stuff_: tuple ):
    y, y_train, pred_train, pred_test, scaler_y, scaler_x= stuff_

    if not arcsined:
        y = scaler_y.inverse_transform(y)
        y_train = scaler_y.inverse_transform(y_train)
        pred_test = scaler_y.inverse_transform(pred_test)

    if not full_pic:
        crashes_utils.prediction_plot(pd.DataFrame(y), pd.DataFrame(y_train), pd.DataFrame(pred_test))
    else:
        if not arcsined:
            pred_train = scaler_y.inverse_transform(pred_train)

        pred = np.concatenate((pred_train, pred_test), axis=0)
        import seaborn as sns
        sns.set_style("whitegrid")
        crashes_utils.seaborn_plot(y, y_train, dates, pred)

    return y, pred_train, pred_test


### initial weights&biases distributions

nor = inits.RandomNormal(mean=0.0, stddev=1.0, seed=2024)
uni = inits.RandomUniform(minval=0.0, maxval=1.0, seed=2024)
truncated_normal = inits.TruncatedNormal(mean=0.0, stddev=0.05, seed=2024)
glorot = inits.GlorotNormal(seed=2024)
glorot_uniform = inits.GlorotUniform(seed=2024)
he_normal = inits.HeNormal(seed=2024)
he_uniform = inits.HeUniform(seed=2024)
orthogonal = inits.Orthogonal(gain=1.0, seed=2024)
constant_ = inits.Constant(1.)
variance_scaling = inits.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform', seed=2024)

### optimizers to use for the networks compilation

adlgrd = opts.Adagrad(learning_rate=0.001, )

adam = opts.Adam(learning_rate=0.006, )
adam_decay = opts.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                                                    decay_steps=500, decay_rate=0.9), )

nadam = opts.Nadam(learning_rate=0.001, )

adadelta = opts.Adadelta(learning_rate=0.001, )

adamax = opts.Adamax(learning_rate=0.001,)

sgd = opts.SGD(learning_rate=0.003, momentum=0.95, nesterov=True)
sgd_decay = opts.SGD(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                                                  decay_steps=70, decay_rate=0.9))
sgd_scale = tf.keras.mixed_precision.LossScaleOptimizer(sgd)

rms_prop = opts.RMSprop(learning_rate=0.001)



### activation functions

# sigmoid
# tanh
# softsign
# selu
# exponential
# softplus
# relu6
# silu
# hard_sigmoid
# linear
# mish

relu = acts.ReLU(max_value=10, negative_slope=0.5, threshold=0.1, )
elu = acts.ELU(alpha = 1. )
leaky_elu = acts.LeakyReLU(alpha=.25)
softplus_layer = lacts.softplus

