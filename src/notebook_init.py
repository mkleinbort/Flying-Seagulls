import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msgn
from catboost import CatBoostRegressor, Pool

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

pd.set_option('max_columns', 200)

from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        if callable(self.columns):
            self.columns = columns(X)
            
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
        
def get_residuals(model, X, y, fit=False):
    if fit:
        model.fit(X,y)
        
    y_resid = y - model.predict(X)
    return y_resid

class ModelChainer():
    '''This class trains each model on the residual of the previous one.'''
    def __init__(self, models=[]):
        self.models = models
        
    def fit(self, X, y):
        for model in self.models:
            y = get_residuals(model, X, y, fit=True)
            
        return self
    
    def predict(self, X):
        return sum(model.predict(X) for model in self.models)

def get_cat_features(X):
    '''Returns the index of boolan/object columns in X'''
    return [i for i, dtype in enumerate(X.dtypes) if dtype in ['object','bool']]

def get_cat_feature_names(X):
    '''Returns the column names of boolan/object columns in X'''
    return np.take(X.columns, get_cat_features(X))

def parse_data_description(filename='../input/house-prices-advanced-regression-techniques/data_description.txt'):
    '''Returns two dictionaries, column_names and value_mappings.
    
    The `column_name` dict maps the columns to their un-abreviated name.
    The `value_mappings` dict maps the values in each categorical column to their un-abreviated value.
    '''
    column_names = {}
    value_mappings = {}
    current_col = None
    with open(filename, 'r') as file:
        for line in file.readlines():
            if ': ' in line and not line.startswith(' '):
                key, val = line.split(': ')
                column_names[key] = val.strip()
                current_col = key
                value_mappings[current_col] = {}
            elif '\t' in line:
                key, val = line.split('\t', 1)
                if key.strip() != '':
                    value_mappings[current_col][key.strip()] = val.strip()
            else:
                pass
            
    value_mappings['MSSubClass'] = {int(key):val for key,val in value_mappings['MSSubClass'].items()}
                
    return column_names, value_mappings

column_names, value_mappings = parse_data_description()

def get_features(load='train', path='../input/house-prices-advanced-regression-techniques/', 
                 column_names=column_names, value_mappings=value_mappings):
    '''Loads the features of from train.csv or test.csv'''
    
    assert load in ['train', 'test']
    
    data = pd.read_csv(f'{path}{load}.csv')    
        
    fillna_dict = {'Alley':'NotApplicable', 
                   'LotFrontage':0, 
                   'MasVnrType':'None',
                   'MasVnrArea':0, 
                   'GarageYrBlt':-1}
    

    X = (data
         .assign(SalePrice = None) # To drop
         .drop(columns=['Id', 'SalePrice'])
         .fillna(fillna_dict)
         .replace({'MSSubClass':value_mappings['MSSubClass']})
        )
        
    other_fillna = {col:'' if col in np.take(X.columns, get_cat_features(X)) else -1 for col in X.columns}

    X = X.fillna(other_fillna)

    return X

def get_target(path='../input/house-prices-advanced-regression-techniques/train.csv', log_y=True):
    y_sale_price = pd.read_csv(path, usecols=['SalePrice'], squeeze=True)
    
    if log_y:
        y_target = np.log(y_sale_price).rename('LogSalePrice')
    else:
        y_target = y_sale_price

    return y_target

def prepare_submission(y_pred, save=False, filename='submission.csv', log_y=True, path='../outputs/'):
    '''Makes a dataframe of the form Id|SalePrice from the y_pred values on the test.csv features.
    Note: Assumes y_pred needs to be transformed with np.exp'''
    
    id_col = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', usecols=['Id'], squeeze=True)
    
    if log_y:
        sale_price = np.exp(y_pred)
    else:
        sale_price = y_pred 
        
    df = pd.DataFrame({
        'Id':id_col,
        'SalePrice':sale_price
    })
    
    if save:
        df.to_csv(f'{path}{filename}', index=False)
        
    return df
    