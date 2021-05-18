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