import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm

# from tqdm import tqdm_notebook

from itertools import product

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

path = '/home/miguel/Python_Projects/datasets/stock_prices_sample.csv'
data = pd.read_csv(path, index_col=['DATE'], parse_dates=['DATE'])
# print(data.head(10))
# print(data.shape)
# print(data.dtypes)

"""
Concentrate on the New Germany Fund (GF) and the end of day (EOD) information.
Remove unwanted columns, to focus on the stocks closing price.
"""
data = data[data.TICKER != 'GEF']
data = data[data.TYPE != 'Intraday']
drop_cols = [
    'SPLIT_RATIO', 'EX_DIVIDEND', 'ADJ_FACTOR', 
    'ADJ_VOLUME', 'ADJ_CLOSE', 'ADJ_LOW', 'ADJ_HIGH', 
    'ADJ_OPEN', 'VOLUME', 'FREQUENCY', 'TYPE', 'FIGI'
    ]
data.drop(drop_cols, axis=1, inplace=True)
print(data.head(10))

### Exploratory data analysis (EDA)
plt.figure(figsize=(17, 8))
plt.plot(data.CLOSE)
plt.title('Closing price of New Germany Fund Inc (GF)')
plt.ylabel('Closing price ($)')
plt.xlabel('Trading day')
plt.grid(False)
plt.show()


## Moving average function
def plot_moving_average(series, window, plot_intervals=False, scale=1.96):
    """Plot moving average with optional confidence intervals"""
    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
    
    #Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')
            
    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

#Smooth by the previous 5 days (by week)
plot_moving_average(data.CLOSE, window=5)
#Smooth by the previous month (30 days)
plot_moving_average(data.CLOSE, window=30)
#Smooth by previous quarter (90 days)
plot_moving_average(data.CLOSE, window=90, plot_intervals=True)


## Exponential smoothing

def exponential_smoothing(series, alpha):

    result = [series.iloc[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series.iloc[n] + (1 - alpha) * result[n-1])
    return result

def plot_exponential_smoothing(series, alphas):
 
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing")
    plt.grid(True)
    plt.show()

plot_exponential_smoothing(data.CLOSE, [0.05, 0.3])






