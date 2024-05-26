import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

import crashes_utils


def kernels(y, y_train, x_train, y_test, x_test, scaler_y, scaler_x, cross_validation: bool):
    forecast_depth = 4
    model_ = 'KernelRidge'
    model_ = 'SVR'

    ts = crashes_utils.crashes
    y_train = (ts.iloc[:int(crashes_utils.train_portion * ts.shape[0])])['Number of crashes']
    y_test = (ts.iloc[int(crashes_utils.train_portion * ts.shape[0]) - forecast_depth + 1:])['Number of crashes']
    y_train = y_train.to_numpy().reshape(y_train.shape + (1,))
    y_test = y_test.to_numpy().reshape(y_test.shape + (1,))
    y = np.concatenate((y_train[forecast_depth - 1:], y_test[forecast_depth - 1:]), axis=0)

    train_data_formatted = np.array(
        [[j for j in y_train[i:i + forecast_depth]] for i in range(0, len(y_train) - forecast_depth + 1)])[:, :, 0]
    test_data_formatted = np.array(
        [[j for j in y_test[i:i + forecast_depth]] for i in range(0, len(y_test) - forecast_depth + 1)])[:, :, 0]

    x_train, y_train = train_data_formatted[:, :forecast_depth - 1], train_data_formatted[:, [forecast_depth - 1]]
    x_test, y_test = test_data_formatted[:, :forecast_depth - 1], test_data_formatted[:, [forecast_depth - 1]]

    if model_ == 'KernelRidge':
        param_grid = {'alpha': [.4, .5, .55, .6, 1], 'coef0': [0, 1],
                      'gamma': [0.336, 0.345, 0.35, 0.36, .365, .375, 0.4, 0.43, 0.45, .46, 0.47, .475, .48, .49, .50,
                                .505, .51, .515, .521, 1],
                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'degree': [1, 2, 3, 4, 5], }
        gr: GridSearchCV = GridSearchCV(KernelRidge(), param_grid, refit=True, verbose=2)
    else:
        param_grid = {'C': [0.1, 1, 10, 100, 150, 200],
                      'gamma': [0.3, 0.35, .375, 0.4, 0.415, 0.42, 0.43, 0.435, 0.44, 0.45, 0.47],
                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                      'epsilon': [0.04, 0.045, 0.05, 0.055, 0.06, 0.08, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21]}
        gr = GridSearchCV(SVR(), param_grid, refit=True, verbose=2)

    gr.fit(x_train, y_train[:, 0])
    print("best estimator: ", gr.best_estimator_)

    y_train_pred = gr.predict(x_train).reshape(-1, 1)
    y_test_pred = gr.predict(x_test).reshape(-1, 1)
    x = pd.to_datetime(crashes_utils.crashes['Ticks']).dt.date[forecast_depth - 1:]

    return y, y_train, y_train_pred, y_test_pred, scaler_y, scaler_x
