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

# Exploratory data analysis (EDA)

plt.figure(figsize=(17, 8))
plt.plot(data.CLOSE)
plt.title('Closing price of New Germany Fund Inc (GF)')
plt.ylabel('Closing price ($)')
plt.xlabel('Trading day')
plt.grid(False)
plt.show()


