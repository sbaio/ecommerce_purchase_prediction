import argparse
import pickle
import os
from time import time
import pandas as pd
import numpy as np
from itertools import product
import json
from sklearn.model_selection import GroupKFold
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="BoostedTree",  help="Model type to use")
    parser.add_argument("-nfolds", type=int, default=10, help="Number of cross validation folds")
    parser.add_argument("-max_cv_runs", type=int, default=-1, help="after how many cv folds to stop ? if -1 run all folds")
    parser.add_argument("-seed", type=int, default=7, help="Seed for controlling randomness")
    parser.add_argument("-n_estimators", type=int, default=100, help="nb estimators for random Forest and boosted Tree classifier")
    parser.add_argument("-data_frac", type=float, default=1., help="fraction of data to use")
    parser.add_argument("-max_depth", type=int, default=3, help="Max depth for trees")
    parser.add_argument("-verbose", type=int, default=0, help="Verbosity")
    parser.add_argument("-outdir", type=str, default="", help="output directory")

    parser.add_argument("-use_customer_data", type=int, default=1, help="Use customer data ?")
    parser.add_argument("-use_product_data", type=int, default=1, help="Use product data ?")
    parser.add_argument("-use_views_data", type=int, default=1, help="Use views data ?")
    parser.add_argument("-use_purchase_data", type=int, default=1, help="Use purchase data ?")

    parser.add_argument("-max_iter", type=int, default=100, help="Max iter for logistic regression")
    
    args = parser.parse_args()
    if args.max_cv_runs <=0:
        args.max_cv_runs = args.nfolds

    return args

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

def fill_missing_values(features):
    # find columns with missing values
    missing_vals_cols = features.columns[features.isna().any()].tolist()

    if 'country' in missing_vals_cols:
        # fill missing country by adding a new
        features.country.fillna('unknown', inplace=True)
    if 'isFemale' in missing_vals_cols:
        # fill binary variables by sampling from their non missing distribution
        features.isFemale = fill_missing_values_binary_prob(features.isFemale)
    if 'isPremier' in missing_vals_cols:
        features.isPremier = fill_missing_values_binary_prob(features.isPremier)
    if 'yearOfBirth' in missing_vals_cols:
        # fill missing yearOfBirth with mean 
        features.yearOfBirth.fillna(value=features.yearOfBirth.mean(), inplace=True)

    return features

def sample_customers(X, groups, frac=0.1):
    kfold = GroupKFold(n_splits=int(1/frac))
    other_index, frac_index = list(kfold.split(X=X, groups=groups))[0]
    # sample_X = X.iloc[val_index]
    # sample_targets = target.iloc[val_index]
    return frac_index, other_index

def get_purchase_info():
    purchases = pd.read_csv("data/purchases.txt")
    purchases = purchases[purchases['date'] < '2016-12-31T23:59:59'] # consider only dates before january
    # print(purchases.shape)

    # remove customer-product pairs in test dataframe
    test_df = pd.read_csv("data/labels_predict.txt")
    # print(test_df.shape)
    purchases['customer_product'] = purchases['customerId'].astype(str) + '_' + purchases['productId'].astype(str)
    test_df['customer_product'] = test_df['customerId'].astype(str) + '_' + test_df['productId'].astype(str)

    # filter out purchases in the test set
    filtered_purchases = purchases[~purchases.customer_product.isin(test_df.customer_product)].drop(columns=['customer_product'])
    
    filtered_purchases['customerPurchaseCount'] = 1 # add a column to count
    customer_purchase_count = filtered_purchases.groupby('customerId').agg({'customerPurchaseCount':'sum'})
    # print(customer_purchase_count)

    filtered_purchases['productPurchaseCount'] = 1 # add a column to count
    product_purchase_count = filtered_purchases.groupby('productId').agg({'productPurchaseCount':'sum'})
    # print(product_purchase_count)
    
    return customer_purchase_count, product_purchase_count

def process_date_col(col):
    date = pd.to_datetime(col, errors='coerce')
    print(f'Nb nan : {date.dt.month.isna().sum()}, filling with median ...')
    date = date.fillna(date.median())

    return pd.DataFrame({
        # 'year':date.dt.year,
        'month':date.dt.month, # categorical 12
        'week':date.dt.isocalendar().week,# categorical 53
        'day':date.dt.day, # categorical 31
        'dayname':date.dt.day_name(), # categorical 7
    })

def load_training_data(args):
    # load training pairs
    df = pd.read_csv("data/labels_training.txt")
    print("Loaded training pairs ", df.shape)

    # keep a separate validation set
    print("Keeping a separate validation set ")
    val_ind, train_ind = sample_customers(df, df['customerId'], frac=0.1)

    if args.data_frac != 1.:
        print("Keeping only a fraction of the training data ")
        train_df = df.iloc[train_ind]
        frac_ind, dropped_ind = sample_customers(train_df, train_df['customerId'], frac=args.data_frac)
        train_ind = train_ind[frac_ind]
        df = df.iloc[np.concatenate([train_ind, val_ind])]

        train_ind = np.arange(train_ind.shape[0])
        val_ind = np.arange(val_ind.shape[0])+train_ind.shape[0]
        print(f"  Train {train_ind.shape[0]}, val {val_ind.shape[0]}")

    purchased = df['purchased'].astype(int)
    customerId = df['customerId']
    # productId = df['productId']

    # load customer info
    customers = pd.read_csv("data/customers.txt")

    # load product info
    products = pd.read_csv("data/products.txt")
    products = products.drop(columns=['dateOnSite'])
    
    if args.use_purchase_data:
        customer_purchase_count, product_purchase_count = get_purchase_info()

        # add customer purchase info ?
        df = pd.merge(df, customer_purchase_count, left_on=['customerId'], right_on=['customerId'], how='left')
        df.customerPurchaseCount.fillna(0, inplace=True)

        # add product purchase info ?
        df = pd.merge(df, product_purchase_count, left_on=['productId'], right_on=['productId'], how='left')
        df.productPurchaseCount.fillna(0, inplace=True)

    if args.use_customer_data:
        print("Using customers dataframe")
        # print(customers)
        df = pd.merge(df, customers, left_on=['customerId'], right_on=['customerId'], how='left')
        
    if args.use_product_data:
        print("Using products dataframe")
        # print(products)
        df = pd.merge(df, products, left_on=['productId'], right_on=['productId'], how='left')

    if args.use_views_data:
        print("Loading views info ...")
        views = pd.read_csv("data/views.txt")
        views = views.drop(columns=['imageZoom']) # discard imageZoom column since all 0 but 1 value
        aggr_views = views.groupby(['customerId','productId']).sum() # aggregate the views of a customer of a product by summing
        df = pd.merge(df, aggr_views, right_index=True, left_on=['customerId', 'productId'])

    ## preprocessing features
    print("Filling missing values")
    df = fill_missing_values(df)

    df.drop(columns=['purchased', 'customerId', 'productId'], inplace=True)

    # encode categorical variables
    columns = df.columns
    print(columns)
    cols_to_encode = ["country", "brand", "productType", "isFemale", "isPremier", "onSale"]
    cols_to_encode = [k for k in cols_to_encode if k in columns]
    column_trans = make_column_transformer(
        (OneHotEncoder(), cols_to_encode),
        remainder="passthrough",
    )
    print("Fitting column transformer on categorical variables")
    column_trans.fit(df)
    print(df.shape)

    return train_ind, val_ind, df, purchased, customerId, column_trans

def main(args):
    print("=================================")
    print("=================================")
    print(args)

    np.random.seed(args.seed)
    
    print("Load and pre-process data ...")
    train_ind, val_ind, df, purchased, customerId, column_trans = load_training_data(args)

    val_df = df.iloc[val_ind]
    val_target = purchased.iloc[val_ind]
    val_features = column_trans.transform(val_df)

    train_features = df.iloc[train_ind] # will be processed by column_trans later
    train_target = purchased.iloc[train_ind]
    train_customerId = customerId.iloc[train_ind]

    print(f"Train {len(train_target)}, val {len(val_target)}")

    # perform a cross-validation and get a validation score
    print("Starting cross-validation")
    kfold = GroupKFold(n_splits=args.nfolds)

    cv_val_aucs = []
    cv_train_aucs = []
    val_aucs = []
    for i, (cv_train_index, cv_val_index) in enumerate(kfold.split(X=train_features, groups=train_customerId)):
        if i >= args.max_cv_runs:
            continue
        start = time()

        print("==============================")
        # verify that no customer is common between train and val dataframes
        unique_train_customers = train_customerId.iloc[cv_train_index].unique()
        unique_val_customers = train_customerId.iloc[cv_val_index].unique()
        assert not len(set(unique_train_customers).intersection(set(unique_val_customers)))

        cv_train_features = column_trans.transform(train_features.iloc[cv_train_index])
        cv_train_targets = train_target.iloc[cv_train_index]
        cv_val_features = column_trans.transform(train_features.iloc[cv_val_index])
        cv_val_targets = train_target.iloc[cv_val_index]

        # Creating model
        kwargs = {
            'random_state':args.seed,
            'verbose':args.verbose,
        }
        if args.model == 'BoostedTree':
            for k in ['max_depth', 'n_estimators', 'loss', 'learning_rate', 'criterion', 'max_features', 'min_samples_split', 'subsample']:
                if k in args:
                    kwargs[k] = getattr(args, k)
            clf = GradientBoostingClassifier(**kwargs)
            print(clf.loss, clf.max_features, clf.n_estimators)
        elif args.model == 'RandomForest':
            for k in ['max_depth', 'n_estimators', 'loss', 'learning_rate', 'criterion', 'max_features', 'min_samples_split', 'subsample']:
                if k in args:
                    kwargs[k] = getattr(args, k)
            clf = RandomForestClassifier(n_jobs=8, **kwargs)
        elif args.model == 'AdaBoost':
            clf = AdaBoostClassifier(n_estimators=args.n_estimators, random_state=args.seed)
        elif args.model == 'LogisticRegression':
            max_iter = args.max_iter if args.max_iter != -1 else 100
            clf = LogisticRegression(random_state=args.seed, verbose=args.verbose, max_iter=max_iter, solver='sag')
            # clf = AdaBoostClassifier(clf, random_state=args.seed, n_estimators=args.n_estimators)
            clf = BaggingClassifier(clf, n_estimators=args.n_estimators, n_jobs=8, verbose=2)
        elif args.model == 'SVM':
            clf = SVC(probability=True, random_state=args.seed, verbose=2)
            # clf = AdaBoostClassifier(clf, random_state=args.seed, n_estimators=args.n_estimators)
            # clf = BaggingClassifier(clf, n_estimators=args.n_estimators, n_jobs=8, verbose=2)
        elif args.model == 'MLP':
            clf = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=2, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
        else:
            raise ValueError(f"Unkown model {args.model}")

        print("Fitting model ... ")
        clf.fit(cv_train_features, cv_train_targets)

        print("Predicting on train samples ...")
        cv_train_pred = clf.predict_proba(cv_train_features)[:,1]
        cv_train_auc = roc_auc_score(cv_train_targets, cv_train_pred)
        cv_train_aucs.append(cv_train_auc)
        print(f"==> {np.mean(cv_train_aucs):0.4f}±{np.std(cv_train_aucs):0.4f} (cur: {cv_train_auc:0.4f})")

        print("Predicting on val samples ... ")
        cv_val_pred = clf.predict_proba(cv_val_features)[:,1]
        cv_val_auc = roc_auc_score(cv_val_targets, cv_val_pred)
        cv_val_aucs.append(cv_val_auc)
        print(f"==> {np.mean(cv_val_aucs):0.4f}±{np.std(cv_val_aucs):0.4f} (cur: {cv_val_auc:0.4f})")

        print("Predicting on main validation set ... ")
        val_pred = clf.predict_proba(val_features)[:,1]
        val_auc = roc_auc_score(val_target, val_pred)
        val_aucs.append(val_auc)
        print(f"==> {np.mean(val_aucs):0.4f}±{np.std(val_aucs):0.4f} (cur: {val_auc:0.4f})")

        print(f"[{i}/{args.nfolds}] Took {(time()-start):0.3f}s")

        # save model
        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            filename = os.path.join(args.outdir, f'model_{i}.pkl')
            pickle.dump(clf, open(filename, 'wb'))

        
    # predict purchasing probabilities on the test data and save the prediction file
    # test_df = pd.read_csv("data/labels_predict.txt")

    out = {
        'cv_train':np.mean(cv_train_aucs),
        'cv_val':np.mean(cv_val_aucs),
        'val':np.mean(val_aucs)
    }
    return out # average cross validation score and filename of saved test results
    
def run_random_forest():
    args = parse_args()
    args.data_frac = 0.1
    args.model = "RandomForest"
    args.verbose = 2
    args.max_cv_runs = 3
    args.nfolds = 10
    args.n_estimators = 100
    # args.min_samples_split = 5
    # args.max_depth = 10 # 50 huge overfitting
    # args.max_features = 0.2
    # args.min_samples_leaf = 70
    # args.max_depth = 200

    main(args)

def run_mlp():
    args = parse_args()
    args.data_frac = 0.01#0.1
    args.model = "MLP"
    args.verbose = 2
    args.max_cv_runs = 3
    args.nfolds = 10

    main(args)

def run_adaboost():
    args = parse_args()
    args.data_frac = 0.1
    args.model = "AdaBoost"
    args.verbose = 2
    args.max_cv_runs = 3
    args.nfolds = 10

    main(args)

# def run_boosted_trees():
#     args = parse_args()
#     args.data_frac = 0.1
#     args.model = "BoostedTree"
#     args.learning_rate = 0.2
#     args.n_estimators = 200
#     args.loss = "exponential"
#     args.verbose = 2
#     args.max_cv_runs = 1#3
#     args.nfolds = 10

def grid_search():
    args = parse_args()
    args.data_frac = 1.0#0.1#1.0#
    args.verbose = 2
    args.max_cv_runs = 1#3
    args.nfolds = 10

    args.model = "BoostedTree"
    # boosted tree
    args.n_estimators = 200
    args.loss = "exponential"
    args.learning_rate = 0.2
    args.outdir = "out/nestimators_lr_loss_frac01/"; 
    param_ranges = {
        "n_estimators":[200,],
        # "subsample":[1.0, 0.9, 0.7, 0.5],
        # "max_depth":[2, 3, 5,],#, 5, 10, 50
        # "criterion":['friedman_mse'],#, "mse"],
        # "min_samples_split":[10, 100], # 2
    }
    print(param_ranges)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # param_ranges = {
        # "loss":["deviance"],#["deviance", "exponential"],
        # "n_estimators":[10, 50, 100, 200, 300],#[100, 50, 200, 400],50, 100, 250, 500
        # "learning_rate":[0.1],#[0.1, 0.05, 0.2],
        # "subsample":[1.0],#, 0.5],
        # "criterion":['friedman_mse'],#, "mse"],
        # "max_depth":[3],#, 5, 10, 50
        # "min_samples_split":[10, 100], # 2
        # "max_features":['auto', 'sqrt', 'log2'],
    # }
    # default_d = dict([(k,v[0]) for k,v in param_ranges.items()])

    out = {}
    keys = param_ranges.keys()
    for x in product(*param_ranges.values()):
        print(" --> ", x)
        args_d = vars(args)
        d = dict(zip(keys, x))
        args_d.update(d)
        args_ = argparse.Namespace(**args_d)

        res = main(args_)
        s = "_".join([f"{k}:{v}" for (k,v) in d.items()])
        out[s] = res['val'] # x['cv_val']

        with open(args.outdir+"out.json", "w") as f:
            json.dump(out, f)

def data_ablation():
    args = parse_args()
    args.n_estimators = 100
    args.model = "BoostedTree"
    args.nfolds = 10
    args.max_cv_runs = 3
    args.data_frac = 0.1
    args.verbose = 2
    
    out = {}
    outfile = "out/ablation/out.json"
    if os.path.exists(outfile):
        with open(outfile, "r") as f:
            out = json.load(f)

    for use_views_data in [1, 0]:
        args.use_views_data = use_views_data
        for use_product_data in [1, 0]:
            args.use_product_data = use_product_data
            for use_purchase_data in [1, 0]:
                args.use_purchase_data = use_purchase_data
                for use_customer_data in [1, 0]:
                    args.use_customer_data = use_customer_data
                    str_ = f"views_{use_views_data}_product_{use_product_data}_purchase_{use_purchase_data}_customer_{use_customer_data}/"
                    print(str_)
                    if str_ in out:
                        continue
                    args.outdir = "out/ablation/"+str_
                    x = main(args)
                    out[str_] = x['val'] #x['cv_val']

                    with open(outfile, "w") as f:
                        json.dump(out, f)

if __name__ == '__main__':
    # args = parse_args()
    # main(args)

    grid_search()
    # data_ablation()
    # run_random_forest()
    # run_mlp()
    # run_adaboost()