import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import crashes_utils


def test_stationarity(timeseries, window, title):
    # Determing rolling statistics
    rolmean = timeseries.rolling(int(window)).mean()
    rolstd = timeseries.rolling(int(window)).std()

    # Plot rolling statistics:
    plt.plot(timeseries, color='blue', label='Data')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('{0}: Rolling Mean & Standard Deviation'.format(title))
    plt.show()

    # Perform Dickey-Fuller test:
    Augmented_Dickey_Fuller_Test_func(timeseries, "")


def two_plots(modified_ts: pd.Series, ts: pd.Series, title: str):
    plt.plot(ts)
    plt.title(title)
    plt.plot(modified_ts, color='red')
    plt.show()


def fitting(ts_, forecast, title, metric, with_conf_int=False, df_conf=None):
    percent = ''
    if metric == 'MAE':
        loss = crashes_utils.mae
    else:
        if metric == 'MSE':
            loss = crashes_utils.mse
        else:
            if metric == 'RMSE':
                loss = crashes_utils.rmse
            else:
                if metric == 'MAPE':
                    loss = crashes_utils.mape1
                else:
                    raise Exception('Wrong loss function specified')
    
    plt.clf()
    plt.plot(ts_)
    plt.plot(forecast, color='red')
    plt.title((title + '. ' + metric + ' = %.1f' % loss(ts_, forecast)) )
    if with_conf_int:
        plt.plot(df_conf['Upper_bound'], label='Confidence Interval Upper bound ')
        plt.plot(df_conf['Lower_bound'], label='Confidence Interval Lower bound ')
    plt.show()


def Augmented_Dickey_Fuller_Test_func(series, column_name):
    print(f'Results of Dickey-Fuller Test for column: {"Crashes No"}')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', 'No Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    if dftest[1] <= 0.05:
        print("Conclusion:====>")
        print("Reject the null hypothesis")
        print("Data is stationary")
    else:
        print("Conclusion:====>")
        print("Fail to reject the null hypothesis")
        print("Data is non-stationary")
    pass


def decompose(ts, decomposition_period):
    # from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = statsmodels.tsa.seasonal.seasonal_decompose(ts, model='additive', period=decomposition_period)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.title('Seasonal decomposition')
    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


class ARMA:
    dynamic_prediction = False
    average_window: int = 12
    decomposition_period: int = 30
    stat_window: int = 12
    model = None

    def __init__(self):
        pass

    def get_model_results(self, ts_train: pd.DataFrame, order):
        try:
            self._model_results = ARIMA(ts_train, order=order).fit()
            return self._model_results
        except ValueError:
            raise ValueError('Wrong or missed order')

    class UnDiff:
        def __init__(self):
            self.undiff_moyen0 = None
            self.undiff_lower_ci0 = None
            self.undiff_upper_ci0 = None

        @property
        def prediction(self):
            return self._prediction

        @prediction.setter
        def prediction(self, p):
            self._prediction = p

    def make_prediction(self, start, end):
        self._prediction = self._model_results.get_prediction(start=start, end=end, dynamic=self.dynamic_prediction)
        return self._prediction

    def create_prediction(self, ts_train, order: tuple, start, end):
        return self.get_model_results(ts_train, order).get_prediction(start=start, end=end,
                                                                      dynamic=self.dynamic_prediction)

    @staticmethod
    def show_forecast(ts, forecast, title, metric, with_conf_int=False, df_conf=None):
        if metric == 'MAE':
            loss = crashes_utils.mae
        else:
            if metric == 'MSE':
                loss = crashes_utils.mse
            else:
                if metric == 'RMSE':
                    loss = crashes_utils.rmse
                else:
                    if metric == 'MAPE':
                        loss = crashes_utils.mape
                    else:
                        raise Exception('Wrong loss function specified')

        plt.plot(ts)
        plt.plot(forecast, color='red')
        plt.title(title + '. ' + metric + ' = %.1f' % loss(ts, forecast))
        if with_conf_int:
            plt.plot(df_conf['Upper_bound'], label='Confidence Interval Upper bound ')
            plt.plot(df_conf['Lower_bound'], label='Confidence Interval Lower bound ')
        plt.show()
        plt.close()
        plt.cla()
        plt.clf()

    def out_of_sample_prediction(self, df: pd.DataFrame, field: str, order, title: str, undiff=False, restore_f=None):
        df_train, df_test = crashes_utils.splitt(df)
        ts_, ts_train, ts_test = df[field], df_train[field], df_test[field]
        ts_test = ts_test.copy()

        if undiff:
            app = pd.Series(ts_train[-1])
            app.index = ts_train[-1:].index
            ts_test = pd.concat([app, ts_test])
            ts_train = ts_train[:-1]
            ts_train_diff = self.diff(ts_train)
            ts_test_diff = self.diff(ts_test)
            pred = self.create_prediction(ts_train_diff, order, start=ts_test_diff.index[0],
                                          end=ts_test_diff.index[len(ts_test_diff.index) - 1])
            moyen = pred.predicted_mean
            moyen = self.undiff(ts_test, moyen)[1:]
            pred_ci = pred.conf_int()
            lower_ci = pd.DataFrame(pred_ci.iloc[:, [0]])
            upper_ci = pd.DataFrame(pred_ci.iloc[:, [1]])
            lower_ci = self.undiff(ts_test, lower_ci)
            upper_ci = self.undiff(ts_test, upper_ci)
            ts_test = ts_test[1:]
            lower_ci = lower_ci[1:]
            upper_ci = upper_ci[1:]
            pred_ci = pd.DataFrame({'lower': lower_ci, 'upper': upper_ci})
        else:
            pred = self.create_prediction(ts_train, order, start=ts_test.index[0],
                                          end=ts_test.index[len(ts_test.index) - 1])
            pred_ci = pred.conf_int()
            moyen = pred.predicted_mean
        if restore_f is not None:
            df = restore_f(df, 0, len(df))
            moyen = moyen.to_frame()
            moyen.columns = [field]
            moyen = restore_f(moyen, int(crashes_utils.train_portion * len(df)), len(df))
            ts_test = ts_test.to_frame()
            ts_test.columns = [field]
            ts_test = restore_f(ts_test, int(crashes_utils.train_portion * len(df)), len(df))
            pred_ci = None

        ARMA.prediction_pic(df, moyen[:-1], ts_test[:-1], title, pred_ci)

    @staticmethod
    def prediction_pic(ts, moyen, ts_test, title, pred_ci=None):
        ax = ts.plot(label='observed')
        moyen.plot(ax=ax, label='Out of sample forecast', alpha=.7)
        if pred_ci is not None:
            ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Crashes no')
        plt.title(title + '. ' + 'MAPE = %.1f' % crashes_utils.mape1(ts_test, moyen))
        plt.legend()
        plt.show()

    @staticmethod
    def diff(ts):
        ts_diff = ts - ts.shift()
        ts_diff.dropna(inplace=True)
        return ts_diff

    @staticmethod
    def undiff(ts, ts_diff):
        ts_undiff = pd.Series((np.insert(np.array(ts_diff), 0, ts[0])).cumsum())
        ts_undiff.index = ts.index
        return ts_undiff

    def analysis(self, df, field, orders: list = None, restore=None):
        ts = df[field]
        plt.clf()
        plt.plot(ts)
        plt.title('Initial data')
        plt.show()

        ts_train, ts_test = crashes_utils.splitt(ts)

        ''' 
         FIRST SEE IF THE TRAINING PART IS  STATIONARY:
        '''

        test_stationarity(ts_train, self.stat_window, 'Initial data')
        decompose(ts_train, self.decomposition_period)
        # autocorrelation(ts_train)
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        plot_acf(ts_train)
        plt.title('Initial data - ACF')
        plt.show()
        plot_pacf(ts_train)
        plt.title('Initial data - PACF')
        plt.show()

        ''' 
         THEN FIT THE MODEL, MAKE PREDICTION, AND COMPARE IT WITH THE TEST STUFF:
        '''

        fitting(ts, self.get_model_results(ts, orders[0]).fittedvalues, 'Initial data', 'MAPE')
        self.out_of_sample_prediction(ts.to_frame(), field, orders[0], 'Initial data')
        if restore is not None:
            # Finally restore globally!
            self.out_of_sample_prediction(df.copy(), field, orders[0], 'sin applied', False, restore)

        ''' 
         FOR BETTER PREDICTION, WE NEED TO A TRANSFORMATION TO MAKE A TIME SERIES AS STATIONARY
         AS POSSIBLE
         TRERE ARE 3 WAYS TO GO:
        '''

        # 1. DIFFERENCING
        ts_diff = self.diff(ts)
        ts_train_diff, _ = crashes_utils.splitt(ts_diff)

        two_plots(ts_diff, ts, title='... now with DIFFERENCING')
        test_stationarity(ts_diff, self.stat_window, 'DIFFERENCING')
        decompose(ts_diff, self.decomposition_period)
        # autocorrelation(ts_train_diff)
        plot_acf(ts_train_diff)
        plt.title('DIFFERENCING - ACF')
        plt.show()
        plot_pacf(ts_train_diff)
        plt.title('DIFFERENCING - PACF')
        plt.show()

        fitting(ts_diff, self.get_model_results(ts_diff,
                                                orders[1]).fittedvalues, 'Initial data' + ' - differencing',
                'MAE')
        self.out_of_sample_prediction(ts_diff.to_frame(), field, orders[1], 'Differenced data')
        if restore is not None:
            # Finally restore globally!
            self.out_of_sample_prediction(df.copy(), field, orders[1], 'sin applied to restored data', True, restore)

        # 2. MOVING AVERAGE
        moving_avg = ts.rolling(self.average_window).mean()
        two_plots(moving_avg, ts, title='... now with MOVING AVERAGE')
        moving_avg_diff = ts - moving_avg
        moving_avg_diff.dropna(inplace=True)
        test_stationarity(moving_avg_diff, self.stat_window, 'MOVING AVERAGE difference')
        decompose(moving_avg_diff, self.decomposition_period)

        # 3. EXPONENTIALLY WEIGHTED MOVING AVERAGE
        expwighted_avg = ts.ewm(self.average_window).mean()
        two_plots(expwighted_avg, ts, '... now with EXP. W. MOVING AVERAGE')
        expwighted_avg_diff = ts - expwighted_avg
        moving_avg_diff.dropna(inplace=True)
        test_stationarity(expwighted_avg_diff, self.stat_window, 'EXP. W. MOVING AVERAGE difference')
        decompose(expwighted_avg_diff, self.decomposition_period)

class AUTO_ARIMA(ARMA):
    def __init__(self, start_p=1, start_q=1, max_p=7, max_q=7, seasonal=False,
                 d=None, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True):
        super().__init__()
        self._start_p = start_p
        self._start_q = start_q
        self._max_p = max_p
        self._max_q = max_q
        self._seasonal = seasonal
        self._d = d
        self._trace = trace
        self._error_action = error_action
        self._suppress_warnings = suppress_warnings
        self._stepwise = stepwise

    def get_model_results(self, ts_train: pd.Series, order=None):
        return auto_arima(ts_train, start_p=self._start_p, start_q=self._start_q,
                          max_p=self._max_p, max_q=self._max_q, seasonal=self._seasonal,
                          d=self._d, trace=self._trace, error_action=self._error_action,
                          suppress_warnings=self._suppress_warnings, stepwise=self._stepwise).arima_res_


class SARIMA(AUTO_ARIMA):
    def __init__(self, ms, start_p_, start_q, max_p, max_q, seasonal,
                 d, trace, error_action, suppress_warnings, stepwise, start_P, start_Q, max_P, max_Q,
                 max_D, D):
        super().__init__(start_p=start_p_, start_q=start_q, max_p=max_p, max_q=max_q, seasonal=seasonal,
                         d=d, trace=trace, error_action=error_action, suppress_warnings=suppress_warnings,
                         stepwise=stepwise)
        print("self._start_p = ", self._start_p)
        self._start_P = start_P
        self._start_Q = start_Q
        self._max_P = max_P
        self._max_Q = max_Q
        self._max_D = max_D
        self._D = D
        self._ms = ms

    def get_model(self, ts_train: pd.Series, order=None):
        for m in self._ms:
            print("=" * 100)
            print(f' Fitting SARIMA for Seasonal value m = {str(m)}')
            model = auto_arima(ts_train, start_p=self._start_p, start_q=self._start_q,
                               max_p=self._max_p, max_q=self._max_q, seasonal=self._seasonal,
                               start_P=self._start_P, start_Q=self._start_Q, max_P=self._max_P,
                               max_D=self._max_D, max_Q=self._max_Q, m=m, d=self._d, D=self._D,
                               trace=self._trace, error_action=self._error_action,
                               suppress_warnings=self._suppress_warnings, stepwise=self._stepwise)

            print(f'Model summary for  m = {str(m)}')
            print("-" * 100)
            model.summary()
            return model

    def get_model_results(self, ts_train: pd.Series, order=None):
        return self.get_model(ts_train, order).arima_res_

    def analysis1(self, ts):
        ts_train, ts_test = crashes_utils.splitt(ts)
        forecast, conf_int = self.get_model(ts_train, ).predict(n_periods=len(ts_test), return_conf_int=True)
        df_conf = pd.DataFrame(conf_int, columns=["Upper_bound", "Lower_bound"])
        df_conf.index = ts_test.index
        forecast = pd.DataFrame(forecast, columns=["forecast"])
        forecast.index = ts_test.index

        crashes_utils.timeseries_evaluation_metrics_func(ts_test, forecast)

        plt.rcParams["figure.figsize"] = [15, 7]
        plt.plot(ts_train, label="Train ")
        plt.plot(ts_test, label="Test ")
        plt.plot(forecast, label=f"Predicted with m={str(self._ms)} ")
        plt.plot(df_conf["Upper_bound"], label="Confidence Interval Upper bound ")
        plt.plot(df_conf["Lower_bound"], label="Confidence Interval Lower bound ")
        plt.legend(loc="best")
        plt.show()
        print("-" * 100)
        print(f" Diagnostic plot for Seasonal value m = {str(self._ms)}")

        # display(model.plot_diagnostics());

        print("-" * 100)


class Trigo:
    index_max = None
    scale = None

    @staticmethod
    def trigo(row):
        if row.name < Trigo.index_max:
            row['Number of crashes'] = np.arcsin(row['Number of crashes'])
        else:
            row['Number of crashes'] = np.pi - np.arcsin(row['Number of crashes'])
        return row['Number of crashes']

    @staticmethod
    def un_trigo(row):
        if row.name < Trigo.index_max:
            row['Number of crashes'] = int(np.sin(row['Number of crashes']) * Trigo.scale)
        else:
            row['Number of crashes'] = int(np.sin(np.pi - row['Number of crashes']) * Trigo.scale)
        return row['Number of crashes']

    @staticmethod
    def restore(df, start_no, end_no):

        """
        :param df: previously modified dataframe
        :param start_no: number, the first df's element correspond to in the general dataset
        :param end_no: number, the last df's element correspond to in the general dataset
        :return: restored dataframe
        """
        df_ = df.copy()
        df.index = np.arange(start_no, end_no)
        df.apply(Trigo.un_trigo, axis=1)
        df.index = df_.index
        return df


ARMA.average_window = 12
ARMA.decomposition_period = 30
ARMA.stat_window = 12
ARMA.dynamic_prediction = False
ARMA.train_portion = crashes_utils.train_portion

crashes = crashes_utils.crashes.copy()
crashes['Ticks'] = pd.to_datetime(crashes['Ticks']).dt.date
crashes = crashes.set_index('Ticks')
field = 'Number of crashes'
ts = crashes[field]

crashes_arcsin = crashes.copy()
crashes_arcsin.index = np.arange(0, len(ts))
Trigo.scale = crashes_arcsin['Number of crashes'].max()
crashes_arcsin['Number of crashes'] = crashes_arcsin['Number of crashes'] / crashes_arcsin['Number of crashes'].max()
arg_max = pd.Series(crashes_arcsin['Number of crashes'].index[crashes_arcsin['Number of crashes']
                                                              == crashes_arcsin['Number of crashes'].max()]).mean()
Trigo.index_max = int(arg_max)
crashes_arcsin.apply(Trigo.trigo, axis=1)
crashes_arcsin.index = crashes.index

'''
 CHOOSE BETWEEN AR_MA AND AUTO ARIMA AND SARIMA
'''

choice = None
while choice != 1 and choice != 2 and choice != 3 and choice != 4:
    print('PLEASE MAKE A RIGHT CHOICE')
    print('TYPE "1" FOR ARMA OR "2" FOR ARIMA OR "3" FOR AUTO ARIMA OR "4" FOR  SARIMA: ')
    choice = int(input())

if choice == 1:
    orders = [(1, 0, 8), (1, 0, 1)] # here and below orders are set according to ACF-PACF pics
    #  ARMA().analysis(crashes, field, orders)
    ARMA().analysis(crashes_arcsin, field, orders, Trigo.restore)

else:
    if choice == 2:
        orders = [(1, 1, 8), (1, 1, 1)]
        # ARMA().analysis(crashes_arcsin, field, orders, Trigo.restore)
        ARMA().analysis(crashes, field, orders)
    else:
        orders = [(0, 0, 0), (0, 0, 0)]
        if choice == 3:
            # AUTO_ARIMA().analysis(crashes, field, orders)
            AUTO_ARIMA().analysis(crashes_arcsin, field, orders, Trigo.restore)
        else:
            SARIMA_ = SARIMA(
                start_p_=1,
                start_q=1,
                max_p=7,
                max_q=7,
                seasonal=True,
                start_P=1,
                start_Q=1,
                max_P=7,
                max_D=7,
                max_Q=7,
                ms=[1, 4, 7, 12, 52],
                d=None,
                D=None,
                trace=True,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            # SARIMA_.analysis(crashes, field, orders)
            SARIMA_.analysis(crashes_arcsin, field, orders, Trigo.restore)
            # SARIMA_.analysis1(crashes)
