import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import crashes_utils

forecast_depth = 15
model_ = 'KernelRidge'
model_ = 'SVR'

ts = crashes_utils.crashes
y_train = (ts.iloc[:int(crashes_utils.train_portion * ts.shape[0])])['Number of crashes']
train_donnees = pd.DataFrame(y_train)
train_ticks = pd.DataFrame((ts.iloc[:int(crashes_utils.train_portion * ts.shape[0])])['Ticks'])
y_test = (ts.iloc[int(crashes_utils.train_portion * ts.shape[0]) - forecast_depth + 1:])['Number of crashes']
test_donnees = pd.DataFrame(y_test)
test_ticks = pd.DataFrame((ts.iloc[int(crashes_utils.train_portion * ts.shape[0]) - forecast_depth + 1:])['Ticks'])

mmscaler = MinMaxScaler()
train_data = mmscaler.fit_transform(train_donnees)
test_data = mmscaler.fit_transform(pd.DataFrame(test_donnees))
y = np.concatenate((y_train[forecast_depth-1:], y_test[forecast_depth-1:]), axis=0)
y_train = (ts.iloc[:int(crashes_utils.train_portion * ts.shape[0])])['Number of crashes']
y_train = y[:int(crashes_utils.train_portion * len(y))]
train_ticks = pd.DataFrame((ts.iloc[:int(crashes_utils.train_portion * ts.shape[0])])['Ticks'])
y_test = (ts.iloc[int(crashes_utils.train_portion * ts.shape[0]) - forecast_depth + 1:])['Number of crashes']
y_test = y[int(crashes_utils.train_portion * len(y)) - forecast_depth + 1:]
test_ticks = pd.DataFrame((ts.iloc[int(crashes_utils.train_portion * ts.shape[0]) - forecast_depth + 1:])['Ticks'])

train_data_formatted = np.array(
    [[j for j in y_train[i:i + forecast_depth]] for i in range(0, len(y_train) - forecast_depth + 1)])[:, :, 0]
test_data_formatted = np.array(
    [[j for j in y_test[i:i + forecast_depth]] for i in range(0, len(y_test) - forecast_depth + 1)])[:, :, 0]

x_train, y_train = train_data_formatted[:, :forecast_depth - 1], train_data_formatted[:, [forecast_depth - 1]]
x_test, y_test = test_data_formatted[:, :forecast_depth - 1], test_data_formatted[:, [forecast_depth - 1]]


if model_ == 'KernelRidge':
    param_grid = {'alpha': [.4, .5, .55, .6, 1], 'coef0': [0, 1], 'gamma': [0.336, 0.345, 0.35, 0.36, .365, .375, 0.4, 0.43, 0.45, 0.47],
                  'kernel': ['rbf', 'poly', 'sigmoid'],  'degree': [1,2,3,4,5],}
    gr: GridSearchCV = GridSearchCV(KernelRidge(), param_grid, refit=True, verbose=2)
else:
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.3, 0.35, .375, 0.4, 0.43, 0.45, 0.47],
                  'kernel': ['rbf', 'poly', 'sigmoid'], 'epsilon': [0.045, 0.05, 0.055]}
    gr = GridSearchCV(SVR(), param_grid, refit=True, verbose=2)

gr.fit(x_train, y_train[:, 0])
print("best estimator: ", gr.best_estimator_)

y_train_pred = gr.predict(x_train).reshape(-1, 1)
y_test_pred = gr.predict(x_test).reshape(-1, 1)
y_train_pred = mmscaler.inverse_transform(y_train_pred)
y_test_pred = mmscaler.inverse_transform(y_test_pred)
y_train = mmscaler.inverse_transform(y_train)
y_test = mmscaler.inverse_transform(y_test)
y = mmscaler.inverse_transform(y)

df_train = train_ticks[forecast_depth-1::]
df_train['Time'] = train_ticks[forecast_depth-1::]
df_train['Number of crashes'] = y_train
df_train_predict = pd.DataFrame()
df_train_predict['Time'] = train_ticks[forecast_depth-1::]
df_train_predict['Number of crashes'] = y_train_pred
x = pd.to_datetime(crashes_utils.crashes['Ticks']).dt.date
predicted = np.concatenate((y_train_pred, y_test_pred), axis=0)
sns.set_style("whitegrid")
crashes_utils.seaborn_plot(y[forecast_depth-1:], y_train, x[forecast_depth-1:], predicted)



g = sns.lineplot(x="Time", y="Number of crashes", data=df_train, color='b')
g = sns.lineplot(x="Time", y="Number of crashes", data=df_train_predict, color='r')
g.xaxis.set_major_locator(ticker.MultipleLocator(3000))
date_format = mpl.dates.DateFormatter('%Y-%m-%d')
g.xaxis.set_major_formatter(date_format)
mpl.pyplot.xticks(rotation=45, fontweight='light', fontsize=12)
mpl.pyplot.show()
print('Train MAPE: ', crashes_utils.mape(y_train, y_train_pred), '%')

df_test = pd.DataFrame()
df_test['Time'] = test_ticks[forecast_depth-1::]
df_test['Number of crashes'] = y_test

df_test_predict = pd.DataFrame()
df_test_predict['Time'] = test_ticks[forecast_depth-1::]
df_test_predict['Number of crashes'] = y_test_pred

# test accuracy
g = sns.lineplot(x="Time", y="Number of crashes", data=df_train, color='k')
g = sns.lineplot(x="Time", y="Number of crashes", data=df_test, color='b').set(title='Test accuracy')
g = sns.lineplot(x="Time", y="Number of crashes", data=df_test_predict, color='r')

mpl.pyplot.show()
print('Test MAPE: ', crashes_utils.mape(y_test, y_test_pred), '%')
