import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def process_date_col(col):
    """
    Given a datetime column col, create columns for categorical values year, month, weekday ...
    """
    date = pd.to_datetime(col, errors='coerce')
    # print(f'Nb nan : {date.dt.month.isna().sum()}, filling with median ...')
    # date = date.fillna(date.median())
    name = col.name
    return pd.DataFrame({
        f'{name}_year':date.dt.year,
        f'{name}_month':date.dt.month, # categorical 12
        # f'{name}_week':date.dt.isocalendar().week,# categorical 53
        f'{name}_day':date.dt.day, # categorical 31
        f'{name}_dayname':date.dt.day_name(), # categorical 7
    })

def sample_customers(X, groups, frac=0.1):
    """
    Sample a train/val split given a fraction, by keeping rows with same group together (same customers)
    """
    kfold = GroupKFold(n_splits=int(1/frac))
    other_index, frac_index = list(kfold.split(X=X, groups=groups))[0]
    return frac_index, other_index

def fill_missing_values_binary_prob(col):
    """
    Fills the column missing value with the same distribution as the non missing ones
    """
    col_ = col.copy()
    d = col_.value_counts().to_dict()
    p = d[True]/(d[True]+d[False]) # proba of being 1
    mask = col_.isna()
    ind = col_.loc[mask].sample(frac=p).index
    col_.loc[ind] = 1
    col_.fillna(0, inplace=True)
    return col_

def fill_missing_values(df):
    # find columns with missing values
    missing_vals_cols = df.columns[df.isna().any()].tolist()
    print(missing_vals_cols)
    if 'country' in missing_vals_cols:
        # fill missing country by adding a new
        df.country.fillna('unknown', inplace=True)
    if 'isFemale' in missing_vals_cols:
        # fill binary variables by sampling from their non missing distribution
        df.loc[:, ('isFemale')] = fill_missing_values_binary_prob(df.isFemale)
    if 'isPremier' in missing_vals_cols:
        df.loc[:, ('isPremier')] = fill_missing_values_binary_prob(df.isPremier)
    if 'yearOfBirth' in missing_vals_cols:
        # fill missing yearOfBirth with mean 
        df.yearOfBirth.fillna(value=df.yearOfBirth.mean(), inplace=True)
    
    if 'dateOnSite_year' in missing_vals_cols:
        df.dateOnSite_year.fillna(df.dateOnSite_year.median(), inplace=True)

    if 'dateOnSite_month' in missing_vals_cols:
        df.dateOnSite_month.fillna(df.dateOnSite_month.median(), inplace=True)

    if 'dateOnSite_day' in missing_vals_cols:
        df.dateOnSite_day.fillna(df.dateOnSite_day.median(), inplace=True)
    return df