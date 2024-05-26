import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, ticker
import seaborn as sns


def mape(predictions, actuals):
    actuals = np.array(actuals)
    try:
        ret = (np.absolute(np.array(predictions) - actuals) / actuals).mean() * 100
    except ZeroDivisionError:
        raise ZeroDivisionError('Divide by 0 in MAPE')
    return ret


# differs from mape by handling zero predictions&actuals
def mape1(predictions, actuals):
    actuals = np.array(actuals)
    i = 0
    ret = 0
    for prediction in np.array(predictions):
        if actuals[i] == 0:
            if prediction != 0:
                ret = ret + np.absolute((prediction - actuals[i]) / prediction)
                i = i + 1
        else:
            ret = ret + np.absolute((prediction - actuals[i]) / actuals[i])
            i = i + 1
    return ret / i * 100


def mae(predictions, actuals):
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    return (np.absolute(predictions - actuals)).mean()


def mse(predictions, actuals):
    return ((np.array(predictions) - np.array(actuals)) ** 2).mean()


def rmse(predictions, actuals):
    return np.sqrt(mse(predictions, actuals))


def prediction_plot(y, y_train, predicted):
    y.index = np.arange(0, len(y))
    y_train.index = np.arange(0, len(y_train))
    predicted.index = np.arange(len(y_train), len(y_train) + len(predicted))
    plt.plot(y)
    plt.plot(predicted, c="g")
    plt.axvline(y_train.index[-1], c="r")
    plt.show()


def seaborn_plot(y: np.array, y_train: np.array, x: np.array, predicted: np.array):
    # bug fix: making x as long as y and predicted because lstm_stateful.LSTM_batch
    # can sometimes shrink them
    x = x[:len(y)]

    df = pd.DataFrame()
    df['Time'] = x
    df['Number of crashes'] = y

    df_predicted = pd.DataFrame()
    df_predicted['Time'] = x
    df_predicted['Number of crashes'] = predicted

    g = sns.lineplot(x="Time", y="Number of crashes", data=df, color='k')
    g.xaxis.set_major_locator(ticker.MultipleLocator(3000))
    date_format = mpl.dates.DateFormatter('%Y-%m-%d')
    g.xaxis.set_major_formatter(date_format)
    (sns.lineplot(x="Time", y="Number of crashes", data=df_predicted, color='y')
     .set(title='Total MAPE' + ' = %.1f' % mape1(y.reshape(y.shape[0]), predicted.reshape(predicted.shape[0])) + ' %'))
    sns.set_style("whitegrid")
    mpl.pyplot.xticks(rotation=45, fontweight='light', fontsize=12)
    mpl.pyplot.axvline(x[len(y_train)], c="r")

    mpl.pyplot.show()


def prediction_plot1(y, y_train, predicted):
    y.index = np.arange(0, len(y))
    y_train.index = np.arange(0, len(y_train))
    predicted.index = np.arange(0, len(predicted))
    plt.plot(y)
    plt.plot(predicted, c="g")
    plt.axvline(y_train.index[-1], c="r")
    plt.title('Total MAPE' + ' = %.1f' % mape1(y[0], predicted[0]) + ' %%')
    plt.show()


def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('Evaluation metric results:-')
    from sklearn import metrics
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MSE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}', end='\n\n')


def splitt(arg):
    return arg[:int(train_portion * len(arg))], arg[
                                                int(train_portion * len(arg)):]


def autocorrelation(ts):
    from statsmodels.tsa.stattools import acf, pacf
    lag_acf = acf(ts, nlags=10)
    lag_pacf = pacf(ts, nlags=10, method='ols')
    # Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')
    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


class Crashes_load:
    dflt: str = '1964-11-24'

    def __init__(self, periods_no, period, throw_start, throw_end, with_c_time=True, ):
        self.with_c_time = with_c_time
        self.crashes = pd.read_csv('Data\\Airplane_Crashes_and_Fatalities_Since_1908.csv')
        idx = pd.Index(self.crashes['Time'])
        Ig = idx.isna()
        self.crashes.insert(1, 'WithoutTime', Ig)
        self.crashes['Time'] = self.crashes.apply(lambda row: self.time_c(row), axis=1)

        # For the case we only want to work with correctly spelled times:
        self.crashes['Date'] = pd.to_datetime(self.crashes['Date'])
        self.crashes['Time'] = pd.to_datetime(self.crashes['Time'], infer_datetime_format=True)
        self.crashes['Time'] = self.crashes.apply(lambda row: self.d_t_combine(row), axis=1)
        self.crashes.drop(["WithoutTime"], axis=1, inplace=True)
        self.crashes = self.crashes[self.throw_interval(self.crashes, 'Date', throw_start, throw_end)]
        ticks_frame = self.prepare_time_frames(self.crashes, periods_no, period, throw_start, throw_end)

        self.crashes['Ticks'] = pd.cut(self.crashes['Time'], ticks_frame['Ticks'], right=True)
        self.crashes = self.crashes.groupby('Ticks').agg({'Time': 'count'}).rename(
            columns={'Time': 'Number of crashes'})
        self.crashes['Ticks'] = self.crashes.index
        self.crashes['Ticks'] = self.crashes['Ticks'].apply(lambda row: row.right)
        self.crashes.index.name = ''

    @staticmethod
    def throw_interval(df, field, throw_start, throw_end):
        return np.logical_or(df[field] <= pd.to_datetime(throw_start), df[field] > pd.to_datetime(throw_end))

    @classmethod
    def prepare_time_frames(self, crashes_, periods_no, period, throw_start, throw_end):
        pass

    class Ticks:
        def __init__(self, points, interval, intervals_no, throw_start, throw_end, with_c_time=True):
            self.with_c_time = with_c_time
            points_no = (points.max() - points.min()).days
            days_in_a_week = 7

            if interval == 'd' or interval == 'D':
                delta = lambda x: datetime.timedelta(days=x)
                interval_in_days = intervals_no
                ticks_no = points_no // intervals_no
            else:
                if interval == 'w' or interval == 'W':
                    delta = lambda x: datetime.timedelta(weeks=x / days_in_a_week)
                    interval_in_days = days_in_a_week * intervals_no
                    ticks_no = points_no // intervals_no // days_in_a_week
                else:
                    if interval == 'q' or interval == 'Q':
                        days_in_a_quarter = days_in_a_week * 31 * 4
                        interval_in_days = days_in_a_quarter * intervals_no
                        delta = lambda x: datetime.timedelta(quarters=x / (days_in_a_week * 31 * 4))
                        ticks_no = points_no // intervals_no // days_in_a_quarter
                    else:
                        raise ValueError('Wrong time interval specified')

            ticks_list = [points.min() + delta(x) for x in range(intervals_no, points_no, interval_in_days)]
            ticks_list.append(points.max())
            self.ticks_frame = pd.DataFrame({'Ticks': ticks_list})
            self.ticks_frame = self.ticks_frame[
                Crashes_load.throw_interval(self.ticks_frame, 'Ticks', throw_start, throw_end)]

    @classmethod
    def time_c(cls, r):
        afternoon_ = '12:00'
        if r['WithoutTime']:
            r['Time'] = afternoon_
        else:
            if r['Time'] == '0943':
                r['Time'] = '09:43'
            if r['Time'] == '114:20':
                r['Time'] = '14:20'
            if r['Time'] == '18.40':
                r['Time'] = '18:40'
            if cls.with_c_time:
                if "'" in r['Time']:
                    l = list(r['Time'])
                    l[2] = ':'
                    emp = ""
                    emp = emp.join(l)
                    r['Time'] = emp
                if r['Time'][0] == 'c':
                    if r['Time'][:2:] == 'c:':
                        if r['Time'][:3:] == 'c: ':
                            r['Time'] = r['Time'][3::]
                        else:
                            r['Time'] = r['Time'][2::]
                    else:
                        r['Time'] = r['Time'][1::]
        return r['Time']

    @classmethod
    def date_time(cls, r):
        try:
            t = pd.to_datetime(pd.Timestamp(r['Time'])).time()
            d = pd.to_datetime(r['Date'])
            return pd.Timestamp.combine(d, t)
        except:
            raise ValueError('Cannot combine data')

    @classmethod
    def d_t_combine(cls, r):
        try:
            return pd.Timestamp.combine(r['Date'], (r['Time']).time())
        except:
            raise ValueError('Cannot combine data')


class Crashes_prepare_date_range(Crashes_load):
    with_c_time = True

    def __init__(self, periods_no, period, with_c_time=True, throw_start=Crashes_load.dflt, throw_end=Crashes_load.dflt):
        self.with_c_time = with_c_time
        super().__init__(periods_no, period, throw_start, throw_end, with_c_time)

    @classmethod
    def prepare_time_frames(cls, crashes, periods_no, period, throw_start, throw_end):
        ret = pd.DataFrame(
            {'Ticks': pd.date_range(pd.to_datetime(crashes['Date']).min(),
                                    pd.to_datetime(crashes['Date']).max(),
                                    freq=str(periods_no) + period)})
        ret = ret[super().throw_interval(ret, 'Ticks', throw_start, throw_end, )]
        return ret


class Crashes_prepare_date_range_business_days_only(Crashes_load):
    with_c_time = True

    def __init__(self, periods_no, period, with_c_time=True, throw_start=Crashes_load.dflt, throw_end=Crashes_load.dflt):
        self.with_c_time = with_c_time
        super().__init__(periods_no, period, throw_start, throw_end, with_c_time)

    @classmethod
    def prepare_time_frames(cls, crashes, periods_no, period, throw_start, throw_end):
        try:
            ret = pd.DataFrame(
                {'Ticks': pd.date_range(pd.to_datetime(crashes['Date']).min(),
                                        pd.to_datetime(crashes['Date']).max(),
                                        freq=str(periods_no) + 'B' + period + 'S')})
        except ValueError:
            raise ValueError('Wrong period specification in the freq parameter')

        ret = ret[super().throw_interval(ret, 'Ticks', throw_start, throw_end, )]
        return ret


class Crashes_prepare_timedelta(Crashes_load):
    with_c_time = True

    def __init__(self, periods_no, period, with_c_time=True, throw_start=Crashes_load.dflt, throw_end=Crashes_load.dflt):
        self.with_c_time = with_c_time
        super().__init__(periods_no, period, throw_start, throw_end, with_c_time)

    @classmethod
    def prepare_time_frames(self, crashes, intervals_no, interval, throw_start, throw_end):
        return self.Ticks(pd.to_datetime(crashes['Date']), interval, intervals_no, throw_start, throw_end).ticks_frame


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


def multi_steps(x_test, x_train, y, y_train, more_time_steps=5):
    assert more_time_steps < len(x_train + 1)
    if not more_time_steps:
        return y, y_train, x_train, x_test

    # [examples, features,  time steps].
    x_train = (np.array(x_train)).reshape(len(x_train), 1)
    x_train_multi = pd.DataFrame(x_train[:, 0])
    for i in np.arange(0, more_time_steps):
        x_train_multi = pd.concat([x_train_multi, pd.DataFrame(x_train).shift(i + 1)], axis=1)
    x_train_multi = x_train_multi.dropna()
    x_train_multi = np.array(x_train_multi).reshape(x_train_multi.shape[0], x_train.shape[1], more_time_steps + 1)
    y_train = y_train[more_time_steps:, ]

    x_test = (np.array(x_test)).reshape(len(x_test), 1)
    x_test_multi = pd.DataFrame(x_test)
    shifted = pd.DataFrame(x_test).shift(1)
    for i in np.arange(1, more_time_steps):
        shifted[0][0] = x_train[-i]
        x_test_multi = pd.concat([x_test_multi, shifted], axis=1)
        shifted = shifted.shift(1)
    shifted[0][0] = x_train[-more_time_steps]
    x_test_multi = pd.concat([x_test_multi, shifted], axis=1)
    x_test_multi = np.array(x_test_multi).reshape(x_test_multi.shape[0], x_test.shape[1], more_time_steps + 1)

    return y[more_time_steps:], y_train, x_train_multi, x_test_multi


# crashes = Crashes_prepare_date_range(4 * 6, 'W', True, '1939-9-1', '1939-9-1').crashes
# train_portion = .8

crashes = Crashes_prepare_timedelta(4 * 6, 'W', True, '1939-9-1', '1939-9-1').crashes
train_portion = .8

# crashes = Crashes_prepare_timedelta(4 * 10, 'W', True, '1939-9-1', '1939-9-1').crashes
# train_portion = .8

# crashes = Crashes_prepare_timedelta(4 * 12, 'W', True, '1939-9-1', '1939-9-1').crashes
# train_portion = .9





