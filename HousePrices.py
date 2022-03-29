import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('C:/Users/aaronk/Desktop/Misc/Dataquest/Real Estate/House_Price_Prediction/Real_Estate_Data.csv',index_col=0)



head = data.head()
info = data.info()
describe = data.describe()

data.hist(bins=40, figsize=(15,10))
# plt.show()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] 

train_set, test_set = split_train_test(data, 0.4)

len(train_set)
len(test_set)

data['LSTAT'].hist(bins=35,figsize=(7,5))
# plt.show()

corr_matrix = data.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

attributes = ['MEDV', 'LSTAT', 'RM']
scatter_matrix(data[attributes], figsize=(12,8))
plt.show()

data['social_cat'] = pd.cut(data['LSTAT'],
                           bins = [0.,10.,20.,30.,40.,np.inf],
                           labels = [1,2,3,4,5])
data['social_cat'].hist()

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.4, random_state = 42)
for train_index, test_index in split.split(data,data['social_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index] 

# strat_test_set['social_cat'].value_counts() / len(strat_test_set)

# for set_ in (strat_train_set, strat_test_set):
#     set_.drop('social_cat', axis=1, inplace=True)

# data.plot(kind='scatter', x='LSTAT',y='MEDV',alpha=0.1)

# data_x = strat_train_set.drop('MEDV', axis=1)
# data_y = strat_train_set['MEDV'].copy()

# imputer = SimpleImputer(strategy='median')
# imputer.fit(data_x)
# data_x_fill = imputer.transform(data_x)

# df_x_fill = pd.DataFrame(data_x_fill, columns = data_x.columns, index = data_x.index)

# forest_reg = RandomForestRegressor()
# forest_reg.fit(data_x_fill, data_y)

# RandomForestRegressor()

# forest_scores = cross_val_score(forest_reg, data_x_fill, data_y, scoring ='neg_mean_squared_error', cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# forest_rmse_scores.mean()

# forest_scores = cross_val_score(forest_reg, data_x_fill, data_y, scoring ='neg_mean_squared_error', cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# forest_rmse_scores.mean()

# data_test_X = strat_test_set.drop('MEDV', axis=1)
# data_test_y = strat_test_set['MEDV'].copy()

# imputer = SimpleImputer(strategy='median')
# imputer.fit(data_test_X)
# data_test_x_fill = imputer.transform(data_test_X)

# final_predictions = forest_reg.predict(data_test_x_fill)

# final_mse = mean_squared_error(data_test_y, final_predictions)
# np.sqrt(final_mse)
# Python C:/Users/aaronk/Desktop/Misc/Dataquest/House_Price_Prediction/HousePrices.py