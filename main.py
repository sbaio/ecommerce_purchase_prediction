import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="RandomForest", choices=['RandomForest', 'BoostedTree', 'LogistiRegression'], help="Model type to use")
    parser.add_argument("-nfolds", type=int, default=10, help="Number of cross validation folds")
    parser.add_argument("-seed", type=int, default=7, help="Seed for controlling randomness")

    args = parser.parse_args()
    return args

def train(train_dset, args):
    model = 1 # create model given args.model
    return model

def predict(model, val_dset):
    score = 1
    return score

def fill_missing_values_binary_prob(col):
    """
    Fills the column missing value with the same distribution as the non missing ones
    """
    d = col.value_counts().to_dict()
    p = d[True]/(d[True]+d[False]) # proba of being 1
    mask = col.isna()
    ind = col.loc[mask].sample(frac=p).index
    col.loc[ind] = 1
    col = col.fillna(0)
    return col

def fill_missing_values(features):
    # find columns with missing values
    missing_vals_cols = features.columns[features.isna().any()].tolist()

    if 'country' in missing_vals_cols:
        # fill missing country by adding a new
        features.country = features.country.fillna('unknown')
    if 'isFemale' in missing_vals_cols:
        # fill binary variables by sampling from their non missing distribution
        features.isFemale = fill_missing_values_binary_prob(features.isFemale)
    if 'isPremier' in missing_vals_cols:
        features.isPremier = fill_missing_values_binary_prob(features.isPremier)
    if 'yearOfBirth' in missing_vals_cols:
        # fill missing yearOfBirth with mean 
        features.yearOfBirth.fillna(value=features.yearOfBirth.mean(), inplace=True)

    return features

def load_training_data():
    # load data
    df = pd.read_csv("data/labels_training.txt")
    views = pd.read_csv("data/views.txt")
    views = views.drop(columns=['imageZoom']) # discard imageZoom column since all 0 but 1 value
    aggr_views = views.groupby(['customerId','productId']).sum() # aggregate the views of a customer of a product by summing

    # add customer info
    customers = pd.read_csv("data/customers.txt")
    df = pd.merge(df, customers, left_on=['customerId'], right_on=['customerId'], how='left')
    # add product info
    products = pd.read_csv("data/products.txt")
    products = products.drop(columns=['dateOnSite'])
    df = pd.merge(df, products, left_on=['productId'], right_on=['productId'], how='left')
    # add views info
    df = pd.merge(df, aggr_views, right_index=True, left_on=['customerId', 'productId'])

    print("Dropping all rows with nan")
    df.dropna(inplace=True) # TODO: temporarily remove all rows with missing info
    print(df.shape)
    # print("Sampling 0.1 fraction of rows")
    # df = df.sample(frac=0.1)
    # print(df.shape)

    target = df['purchased'].astype(int)
    customerId = df['customerId']
    # productId = df['productId']
    features = df.drop(columns=['purchased', 'customerId', 'productId'])

    return features, target, customerId

def main():
    args = parse_args()
    print(args)

    np.random.seed(args.seed)
    
    print("Loading data ...")
    features, target, customerId = load_training_data()

    ## preprocessing features
    # fill missing values
    print("Filling missing values")
    features = fill_missing_values(features)

    # encode categorical variables # TODO: add categorical variables
    print("Encoding categorical variables")
    features = features.drop(columns=['country', 'productType', 'brand']) 

    # country_enc = OneHotEncoder()
    # features.country = country_enc.fit_transform(features.country.values)

    # product type

    features.isFemale = features.isFemale.astype(int)
    features.isPremier = features.isPremier.astype(int)
    features.onSale = features.onSale.astype(int)

    # perform a cross-validation and get a validation score
    print("Starting cross-validation")
    kfold = GroupKFold(n_splits=args.nfolds)
    # kfold = GroupShuffleSplit(n_splits=args.nfolds, test_size=0.1, random_state=args.seed)

    aucs = []
    for train_index, val_index in kfold.split(X=features, groups=customerId):
        # print(train_index[:3], val_index[:3])
        # print(train_index.shape, val_index.shape)

        # # verify that no customer is common between train and val dataframes
        unique_train_customers = customerId.iloc[train_index].unique()
        unique_val_customers = customerId.iloc[val_index].unique()
        assert not len(set(unique_train_customers).intersection(set(unique_val_customers)))

        train_features = features.iloc[train_index]
        train_targets = target.iloc[train_index]
        val_features = features.iloc[val_index]
        val_targets = target.iloc[val_index]

        clf = RandomForestClassifier(random_state=args.seed)
        print("Fitting model ... ")
        clf.fit(train_features, train_targets)
        print("Predicting on val samples ... ")
        val_pred = clf.predict_proba(val_features)[:,1]

        auc = roc_auc_score(val_targets, val_pred)
        print(auc)
        aucs.append(auc)
        # for each row, append the customer info, product info, and the views info of customer and product
    print(np.mean(aucs), np.std(aucs))


    # use ensembling (bagging)
    # predict purchasing probabilities on the test data and save the prediction file
    test_df = pd.read_csv("data/labels_predict.txt")

    return 1 # average cross validation score and filename of saved test results
    

if __name__ == '__main__':
    main()