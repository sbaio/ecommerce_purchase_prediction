import argparse
from time import time
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupKFold
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="RandomForest", 
                            choices=['RandomForest', 'BoostedTree', 'AdaBoost', 'LogisticRegression', 'RandomForest_seippel'], 
                            help="Model type to use")
    parser.add_argument("-nfolds", type=int, default=10, help="Number of cross validation folds")
    parser.add_argument("-max_cv_runs", type=int, default=-1, help="after how many cv folds to stop ? if -1 run all folds")
    parser.add_argument("-seed", type=int, default=7, help="Seed for controlling randomness")
    parser.add_argument("-n_estimators", type=int, default=100, help="nb estimators for random Forest and boosted Tree classifier")
    parser.add_argument("-data_frac", type=float, default=1., help="fraction of data to use")
    parser.add_argument("-max_depth", type=int, default=3, help="Max depth for gradient boosting")
    parser.add_argument("-verbose", type=int, default=0, help="Verbosity")
    parser.add_argument("-drop_nan", type=int, default=1, help="Drop rows with nan")
    
    args = parser.parse_args()
    return args

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

def sample_customers(X, target, groups, frac=0.1):
    kfold = GroupKFold(n_splits=int(1/frac))
    train_index, val_index = list(kfold.split(X=X, groups=groups))[0]
    sample_X = X.iloc[val_index]
    sample_targets = target.iloc[val_index]
    return sample_X, sample_targets, val_index

def load_training_data(args):
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

    print(df.shape)
    if args.drop_nan:
        print("Dropping all rows with nan")
        df.dropna(inplace=True) # TODO: temporarily remove all rows with missing info
        print(df.shape)
    
    frac = args.data_frac
    if frac != 1.:
        print(f"Sampling {frac} fraction of rows")
        # df = df.sample(frac=frac) # sample randomly
        df, _, sampled_inds = sample_customers(df, df['purchased'], df['customerId'], frac=frac) # sample by grouping customers
        print(df.shape)

    target = df['purchased'].astype(int)
    customerId = df['customerId']
    # productId = df['productId']
    features = df.drop(columns=['purchased', 'customerId', 'productId'])

    return features, target, customerId

def main(args):
    print(args)

    np.random.seed(args.seed)
    
    print("Loading data ...")
    features, target, customerId = load_training_data(args)

    ## preprocessing features
    print("Filling missing values")
    features = fill_missing_values(features)

    # encode categorical variables
    column_trans = make_column_transformer(
        (OneHotEncoder(), ["country", "brand", "productType", "isFemale", "isPremier", "onSale"]),
        remainder="passthrough",
    )
    print("Fitting column transformer on categorical variables")
    column_trans.fit(features)

    # perform a cross-validation and get a validation score
    print("Starting cross-validation")
    kfold = GroupKFold(n_splits=args.nfolds)

    aucs = []
    for i, (train_index, val_index) in enumerate(kfold.split(X=features, groups=customerId)):
        if args.max_cv_runs >0 and i >= args.max_cv_runs:
            continue
        start = time()

        # verify that no customer is common between train and val dataframes
        unique_train_customers = customerId.iloc[train_index].unique()
        unique_val_customers = customerId.iloc[val_index].unique()
        assert not len(set(unique_train_customers).intersection(set(unique_val_customers)))

        train_features = column_trans.transform(features.iloc[train_index])
        train_targets = target.iloc[train_index]
        val_features = column_trans.transform(features.iloc[val_index])
        val_targets = target.iloc[val_index]

        kwargs = {
            'random_state':args.seed,
            'verbose':args.verbose,
        }
        
        if args.model == 'RandomForest':
            clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed, verbose=args.verbose)
        elif args.model == 'BoostedTree':
            for k in ['max_depth', 'n_estimators', 'loss', 'learning_rate', 'criterion', 'max_features', 'min_samples_split', 'subsample']:
                if k in args:
                    kwargs[k] = getattr(args, k)
            clf = GradientBoostingClassifier(**kwargs)
            print(clf.loss, clf.max_features, clf.n_estimators)
        elif args.model == 'AdaBoost':
            clf = AdaBoostClassifier(n_estimators=args.n_estimators, random_state=args.seed, verbose=args.verbose)
        elif args.model == 'RandomForest_seippel':
            clf = RandomForestClassifier(n_estimators=400, random_state=args.seed, max_depth=200, verbose=2, min_samples_leaf=70, max_features=0.2)
        elif args.model == 'LogisticRegression': # TODO: use ensembling
            clf = LogisticRegression(random_state=args.seed, verbose=args.verbose)
        else:
            raise ValueError(f"Unkown model {args.model}")

        print("Fitting model ... ")
        clf.fit(train_features, train_targets)
        print("Predicting on train samples ...")
        train_pred = clf.predict_proba(train_features)[:,1]
        train_auc = roc_auc_score(train_targets, train_pred)
        print(train_auc)
        print("Predicting on val samples ... ")
        val_pred = clf.predict_proba(val_features)[:,1]
        val_auc = roc_auc_score(val_targets, val_pred)
        print(val_auc)
        aucs.append(val_auc)
        print(f"Took {(time()-start):0.3f}s")
        
        val_mean, val_std = np.mean(aucs), np.std(aucs)
        print(val_mean, val_std)

    
    # predict purchasing probabilities on the test data and save the prediction file
    # test_df = pd.read_csv("data/labels_predict.txt")

    return [np.mean(aucs), np.std(aucs)] # average cross validation score and filename of saved test results
    
def grid_search():
    args = parse_args()
    args.data_frac = 0.1
    args.model = "BoostedTree"
    args.verbose = 2
    args.drop_nan = 0
    args.max_cv_runs = 3
    
    param_ranges = {
        "loss":["exponential"],#["deviance", "exponential"],
        "n_estimators":[400],#[100, 50, 200, 400],
        "learning_rate":[0.2],#[0.1, 0.05, 0.2],
        "subsample":[1.0],#, 0.5],
        "criterion":['friedman_mse'],#, "mse"],
        "max_depth":[3, 5, 10, 50],
        # "min_samples_split":[2, 5, 8],
        # "max_features":['auto', 'sqrt', 'log2'],
    }
    default_d = dict([(k,v[0]) for k,v in param_ranges.items()])
    
    out = {}
    for param, values in param_ranges.items():
        args_d = vars(args)
        args_d.update(default_d)
        args_ = argparse.Namespace(**args_d)
        out[param] = {}
        for val in values:
            print("====================")
            setattr(args_, param, val)
            
            val_out = main(args_)
            out[param][val] = val_out

            with open("out.json", "w") as f:
                json.dump(out, f)

def train_gbtrees():
    args = parse_args()
    args.n_estimators = 400
    args.learning_rate = 0.2
    args.loss = "exponential"
    args.verbose = 2
    args.model = "BoostedTree"
    args.drop_nan = 0
    args.data_frac = 0.1

    main(args)

if __name__ == '__main__':
    # args = parse_args()
    # main(args)

    grid_search()

    # train_gbtrees()