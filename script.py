import argparse
import pickle
import os
from time import time
import pandas as pd
import numpy as np
from itertools import product
import json
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from util import process_date_col, sample_customers

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-use_customer_data", type=int, default=1, help="Use customer data ?")
    parser.add_argument("-use_product_data", type=int, default=1, help="Use product data ?")
    parser.add_argument("-use_views_data", type=int, default=1, help="Use views data ?")
    parser.add_argument("-use_purchase_data", type=int, default=1, help="Use purchase data ?")

    parser.add_argument("-heldout_frac", type=float, default=0.1, help="fraction of data to use")

    parser.add_argument("-outdir", type=str, default="", help="output directory")

    parser.add_argument("-model", type=str, default="BoostedTree",  help="Model type to use")

    parser.add_argument("-nfolds", type=int, default=5, help="Number of cross validation folds")
    parser.add_argument("-seed", type=int, default=7, help="Seed for controlling randomness")
    
    parser.add_argument("-data_frac", type=float, default=1., help="fraction of data to use")
    parser.add_argument("-max_depth", type=int, default=3, help="Max depth for trees")
    parser.add_argument("-verbose", type=int, default=0, help="Verbosity")
    
    # parser.add_argument("-fill_nan", type=int, default=1, help="Fill NaNs")
    # parser.add_argument("-max_iter", type=int, default=100, help="Max iter for logistic regression")

    # lightgbm params
    parser.add_argument("-n_estimators", type=int, default=100, help="nb estimators for random Forest and boosted Tree classifier")
    parser.add_argument("-num_leaves", type=int, default=31, help="Nb of leaves")
    parser.add_argument("-learning_rate", type=float, default=0.1, help="learning rate")
    parser.add_argument("-boosting_type", type=str, default="gbdt",  help="Type of boosting gbdt, dart, goss, rf")
    
    parser.add_argument("-cv", type=int, default=1, help="perform cross validation or just train and validate on held out set")

    parser.add_argument("-test", type=int, default=0, help="Run test")
    
    parser.add_argument("-task", type=str, default="main",  help="Which fct to run ?")

    args = parser.parse_args()
    if args.test:
        print("test=1 ==> cv=0")
        args.cv = 0
    
    return args

def get_purchase_info():
    purchases = pd.read_csv("data/purchases.txt")
    purchases = purchases[purchases['date'] < '2016-12-31T23:59:59'] # consider only dates before january

    # remove customer-product pairs in test dataframe
    test_df = pd.read_csv("data/labels_predict.txt")
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

def create_model(args):
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

    return clf

def setup_data(args):
    # load from cache ?
    cache_dir = "/mnt/hdd/sbaio/asos/"
    cache_file = cache_dir+f"cus{args.use_customer_data}_prod{args.use_product_data}_view{args.use_views_data}_pur{args.use_purchase_data}_held{args.heldout_frac}.pkl"
    if os.path.exists(cache_file):
        df, label, customerId, train_ind, val_ind, df_test = pickle.load(open(cache_file, "rb"))
        return df, label, customerId, train_ind, val_ind, df_test

    # load training pairs
    df = pd.read_csv("data/labels_training.txt")
    print("Loaded training pairs ", df.shape)

    if args.test:
        # load testing pairs
        df_test = pd.read_csv("data/labels_predict.txt")
        print("Loaded testing pairs ", df_test.shape)
    else:
        df_test = None

    ## use customer data
    if args.use_customer_data:
        customers = pd.read_csv("data/customers.txt")
        # merge customers into df
        df = pd.merge(df, customers, left_on=['customerId'], right_on=['customerId'], how='left')

        # create a mapping for country to id
        countries = df.country.astype('category').cat.categories
        df.loc[:,["country"]] = df.country.astype('category').cat.codes.astype(int)
        
        # pre-process columns
        bool_d = {"True":1, "False":0, '1':1, '0':0, 1:1, 0:0}
        df.isFemale = df.isFemale.map(bool_d, na_action='ignore')
        df.isPremier = df.isPremier.map(bool_d, na_action='ignore')

        if args.test:
            # merge customers into df_test
            df_test = pd.merge(df_test, customers, left_on=['customerId'], right_on=['customerId'], how='left')
            # map countries and map unkown ones to a new class (len(countries))
            country2id = dict([(v,k) for k,v in enumerate(countries)])
            df_test.country = df_test.country.map(country2id).fillna(len(countries)).astype(int)

            df_test.isFemale = df_test.isFemale.map(bool_d, na_action='ignore')
            df_test.isPremier = df_test.isPremier.map(bool_d, na_action='ignore')
        
    ## use product data
    if args.use_product_data:
        products = pd.read_csv("data/products.txt")
        # replace dateOnSite with year, month, day ...
        dateOnSite = process_date_col(products['dateOnSite'])
        products = products.drop(columns=['dateOnSite'])
        products = pd.concat([products, dateOnSite], axis=1)
        # update df
        df = pd.merge(df, products, left_on=['productId'], right_on=['productId'], how='left')

        # pre-process dateOnSite (create a mapping of weekdays)
        days = df.dateOnSite_dayname.astype('category').cat.categories
        df.loc[:,["dateOnSite_dayname"]] = df.dateOnSite_dayname.astype('category').cat.codes.astype(int)
        if args.test:
            df_test = pd.merge(df_test, products, left_on=['productId'], right_on=['productId'], how='left')
            day2id = dict([(v,k) for k,v in enumerate(days)])
            df_test.dateOnSite_dayname = df_test.dateOnSite_dayname.map(day2id).fillna(len(days)).astype(int)

    # add views
    if args.use_views_data:
        print("Loading views info ...")
        views = pd.read_csv("data/views.txt")
        views = views.drop(columns=['imageZoom']) # discard imageZoom since all 0 but 1 value
        aggr_views = views.groupby(['customerId','productId']).sum() # aggregate the views
        # update df
        df = pd.merge(df, aggr_views, right_index=True, left_on=['customerId', 'productId'])
        if args.test:
            df_test = pd.merge(df_test, aggr_views, right_index=True, left_on=['customerId', 'productId'])

    ## add purchase info
    if args.use_purchase_data:
        customer_purchase_count, product_purchase_count = get_purchase_info()
        # add customer purchase info ?
        df = pd.merge(df, customer_purchase_count, left_on=['customerId'], right_on=['customerId'], how='left')
        df.customerPurchaseCount.fillna(0, inplace=True)
        # add product purchase info ?
        df = pd.merge(df, product_purchase_count, left_on=['productId'], right_on=['productId'], how='left')
        df.productPurchaseCount.fillna(0, inplace=True)

        if args.test:
            # add customer purchase info ?
            df_test = pd.merge(df_test, customer_purchase_count, left_on=['customerId'], right_on=['customerId'], how='left')
            df_test.customerPurchaseCount.fillna(0, inplace=True)
            # add product purchase info ?
            df_test = pd.merge(df_test, product_purchase_count, left_on=['productId'], right_on=['productId'], how='left')
            df_test.productPurchaseCount.fillna(0, inplace=True)
        
    # add other features
    # TODO: add other features

    # keep held-out set
    heldout_frac = args.heldout_frac
    val_ind, train_ind = sample_customers(df, df['customerId'], frac=heldout_frac)

    label = df.purchased
    customerId = df.customerId
    df.drop(columns=['purchased', 'customerId', 'productId'], inplace=True)
    if args.test:
        df_test.drop(columns=['purchase_probability', 'customerId', 'productId'], inplace=True)

    out = df, label, customerId, train_ind, val_ind, df_test
    pickle.dump(out, open(cache_file,"wb"))
    return out

def main(args):
    print("=================================")
    print(args)

    np.random.seed(args.seed)

    df, label, customerId, train_ind, val_ind, df_test = setup_data(args)

    param = {
        'num_leaves': args.num_leaves, 
        'objective': 'binary',
        'learning_rate':args.learning_rate,
        'num_leaves':args.num_leaves,
        'num_threads':4,
        'force_row_wise':True,

        'metric':'auc',
    }

    train_df = df.iloc[train_ind]
    val_df = df.iloc[val_ind]
    train_dset = lgb.Dataset(train_df, label=label.iloc[train_ind])
    val_dset = lgb.Dataset(val_df, label=label.iloc[val_ind])

    num_round = args.n_estimators
    if args.cv:
        print(f"Cross validation {args.nfolds} folds")
        kfold = GroupKFold(n_splits=args.nfolds)
        folds = kfold.split(X=df.iloc[train_ind], groups=customerId.iloc[train_ind])
        eval_hist = lgb.cv(param, train_dset, num_round, folds=folds)

        mean = eval_hist['auc-mean'][-1]
        std = eval_hist['auc-stdv'][-1]
        print(f"==> {mean:0.4f}±{std:0.4f}")
        return mean, std
    else:
        print("Training model ")
        bst = lgb.train(param, train_dset, num_round, valid_sets=[val_dset])

        train_pred = bst.predict(train_df)
        train_auc = roc_auc_score(label.iloc[train_ind], train_pred)
        val_auc = bst.best_score['valid_0']['auc']
        print(f"Train {train_auc:0.4f}, Val {val_auc:0.4f}")

    # if outdir provided ?
    if args.test:
        outdir = args.outdir if args.outdir else "./"
        test_pred = bst.predict(df_test)
        df_test = pd.read_csv("data/labels_predict.txt")
        df_test['purchase_probability'] = test_pred
        outfile = outdir+'out.csv'
        df_test.to_csv(outfile)
        print(f"Saved to {outfile}")
    
    return val_auc, train_auc
    
def grid_search(args, param_ranges):
    args.cv = 1
    
    print(param_ranges)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    out = {}
    
    keys = param_ranges.keys()
    for x in product(*param_ranges.values()):
        args_d = vars(args)
        d = dict(zip(keys, x))
        args_d.update(d)
        args_ = argparse.Namespace(**args_d)

        s = "_".join([f"{k}:{v}" for (k,v) in d.items()])
        print(" --> ", s)

        res = main(args_)
        
        out[s] = res[0]

        with open(args.outdir+"out.json", "w") as f:
            json.dump(out, f)

if __name__ == '__main__':
    args = parse_args()
    if args.task == 'main':
        main(args)
    elif args.task == 'grid_search':
        args.outdir = "out/grid_search/"
        param_ranges = {
            "learning_rate":[0.1, 0.2, 0.05],
            "num_leaves":[10, 31, 50, 100],
            'n_estimators':[100, 200, 300],
        }
        grid_search(args, param_ranges)
    elif args.task == "data_ablation":
        args.outdir = "out/data_ablation/"
        param_ranges = {
            'n_estimators':[100],
            "use_views_data":[1, 0],
            "use_product_data":[1, 0],
            "use_purchase_data":[1, 0],
            "use_customer_data":[1, 0],
        }
        grid_search(args, param_ranges)


# TODO:
# implement other models ?
# run data ablation
# run grid search
# add data_frac ?