import argparse
import pickle
import os
import pandas as pd
import numpy as np
from itertools import product
import json
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

from util import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-cv", type=int, default=1, help="perform cross validation or just train and validate on held out set")
    parser.add_argument("-test", type=int, default=0, help="Run inference on test")
    parser.add_argument("-task", type=str, default="main",  help="Which fct to run ?")

    parser.add_argument("-use_customer_data", type=int, default=1, help="Use customer data ?")
    parser.add_argument("-use_product_data", type=int, default=1, help="Use product data ?")
    parser.add_argument("-use_views_data", type=int, default=1, help="Use views data ?")
    parser.add_argument("-use_purchase_data", type=int, default=1, help="Use purchase data ?")

    parser.add_argument("-heldout_frac", type=float, default=0.1, help="fraction of data to use")

    parser.add_argument("-outdir", type=str, default="", help="output directory")

    parser.add_argument("-model", type=str, default="light_gbdt",  help="Model type to use")

    parser.add_argument("-nfolds", type=int, default=5, help="Number of cross validation folds")
    parser.add_argument("-seed", type=int, default=7, help="Seed for controlling randomness")
    
    parser.add_argument("-data_frac", type=float, default=1., help="fraction of data to use")
    parser.add_argument("-max_depth", type=int, default=3, help="Max depth for trees")
    parser.add_argument("-verbose", type=int, default=0, help="Verbosity")

    # lightgbm params
    parser.add_argument("-n_estimators", type=int, default=100, help="nb estimators for random Forest and boosted Tree classifier")
    parser.add_argument("-num_leaves", type=int, default=31, help="Nb of leaves")
    parser.add_argument("-learning_rate", type=float, default=0.1, help="learning rate")
    parser.add_argument("-boosting_type", type=str, default="gbdt",  help="Type of boosting gbdt, dart, goss, rf")

    parser.add_argument("-data_dir", type=str, default="data/",  help="Folder containing customers.txt, products.txt, views.txt ...")
    parser.add_argument("-cache_dir", type=str, default="/mnt/hdd/sbaio/asos/",  help="Folder to save cache files to make loading faster ...")

    

    args = parser.parse_args()
    if args.test:
        print("test=1 ==> cv=0")
        args.cv = 0
    
    return args

def get_purchase_info():
    purchases = pd.read_csv(args.data_dir+"purchases.txt")
    purchases = purchases[purchases['date'] < '2016-12-31T23:59:59'] # consider only dates before january

    # remove customer-product pairs in test dataframe
    test_df = pd.read_csv(args.data_dir+"labels_predict.txt")
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
    kwargs = {
        'random_state':args.seed,
        'verbose':args.verbose,
    }
    if args.model == 'BoostedTree':
        clf = GradientBoostingClassifier(learning_rate=args.learning_rate, n_estimators=args.n_estimators, **kwargs)
    elif args.model == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=args.n_estimators, **kwargs)
    elif args.model == 'AdaBoost':
        clf = AdaBoostClassifier(n_estimators=args.n_estimators, random_state=args.seed)
    elif args.model == 'LogisticRegression':
        max_iter = args.max_iter if args.max_iter != -1 else 100
        clf = LogisticRegression(random_state=args.seed, verbose=args.verbose, max_iter=max_iter, solver='sag')
        # clf = AdaBoostClassifier(clf, random_state=args.seed, n_estimators=args.n_estimators)
        clf = BaggingClassifier(clf, n_estimators=args.n_estimators, n_jobs=8, verbose=2)
    elif args.model == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=2, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
    else:
        raise ValueError(f"Unkown model {args.model}")
    return clf

def setup_data(args):
    # load from cache ?
    cache_file = args.cache_dir+f"cus{args.use_customer_data}_prod{args.use_product_data}_view{args.use_views_data}_pur{args.use_purchase_data}_held{args.heldout_frac}_test{args.test}.pkl"
    if os.path.exists(cache_file):
        df, label, customerId, train_ind, val_ind, df_test = pickle.load(open(cache_file, "rb"))
        return df, label, customerId, train_ind, val_ind, df_test

    # load training pairs
    df = pd.read_csv(args.data_dir+"labels_training.txt")
    print("Loaded training pairs ", df.shape)

    if args.test:
        # load testing pairs
        df_test = pd.read_csv(args.data_dir+"labels_predict.txt")
        print("Loaded testing pairs ", df_test.shape)
    else:
        df_test = None

    ## use customer data
    if args.use_customer_data:
        customers = pd.read_csv(args.data_dir+"customers.txt")
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
        products = pd.read_csv(args.data_dir+"products.txt")
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
        views = pd.read_csv(args.data_dir+"views.txt")
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
    if os.path.exists(args.cache_dir):
        pickle.dump(out, open(cache_file,"wb"))
    return out

def fit_column_transformer(df):
    # encode categorical variables
    columns = df.columns
    print(columns)
    cols_to_encode = ["country", "brand", "productType", "isFemale", "isPremier", "onSale"]
    cols_to_encode = [k for k in cols_to_encode if k in columns]
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), cols_to_encode),
        remainder="passthrough",
    )
    print("Fitting column transformer on categorical variables")
    column_trans.fit(df)
    return column_trans


def main(args):
    print("=================================")
    print(args)

    np.random.seed(args.seed)

    df, label, customerId, train_ind, val_ind, df_test = setup_data(args)
    
    param = {
        'boosting_type':args.boosting_type,
        'is_unbalance':True,
        'num_leaves': args.num_leaves, 
        'objective': 'binary',
        'learning_rate':args.learning_rate,
        'num_leaves':args.num_leaves,
        'num_threads':4,
        'force_row_wise':True,

        'metric':'auc',
    }

    if args.data_frac != 1:
        print(f"Keeping only {args.data_frac} of the training data")
        # sample inds by grouping customers
        frac_ind, dropped_ind = sample_customers(df.iloc[train_ind], customerId.iloc[train_ind], frac=args.data_frac)
        print(f"  From {len(train_ind)} to {len(frac_ind)}")
        train_ind = train_ind[frac_ind]
        
        print(f"  Train {train_ind.shape[0]}, val {val_ind.shape[0]}")

    train_df = df.iloc[train_ind]
    val_df = df.iloc[val_ind]
    train_target = label.iloc[train_ind]
    val_target = label.iloc[val_ind]

    if args.model == "light_gbdt":
        train_dset = lgb.Dataset(train_df, label=train_target)
        val_dset = lgb.Dataset(val_df, label=val_target)

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
            train_auc = roc_auc_score(train_target, train_pred)
            val_auc = bst.best_score['valid_0']['auc']
            print(f"Train {train_auc:0.4f}, Val {val_auc:0.4f}")


            # classification report
            val_pred = bst.predict(val_df)
            print(classification_report(val_target, val_pred > 0.5))
            print(confusion_matrix(val_target, val_pred>0.5))

        if args.test:
            print("Testing ...")
            outdir = args.outdir if args.outdir else "./"
            os.makedirs(outdir, exist_ok=True)
            bst.save_model(outdir + 'model.txt')
            test_pred = bst.predict(df_test)
            df_test = pd.read_csv(args.data_dir+"labels_predict.txt")
            df_test['purchase_probability'] = test_pred
            outfile = outdir+'out.csv'
            df_test.to_csv(outfile)
            print(f"Saved to {outfile}")

            
        
        return val_auc, train_auc
    else:
        clf = create_model(args)

        # fill missing values
        df = fill_missing_values(df)
        train_df = df.iloc[train_ind]
        val_df = df.iloc[val_ind]

        # process df: fill missing values, onehot encode categorical variable
        col_trans = fit_column_transformer(train_df)

        # train on train_df and validate on val_df
        train_feats = col_trans.transform(train_df)
        val_feats = col_trans.transform(val_df)

        print("Fitting model ... ")
        clf.fit(train_feats, train_target)

        print("Predicting on train samples ...")
        train_pred = clf.predict_proba(train_feats)[:,1]
        train_auc = roc_auc_score(train_target, train_pred)
        print(f"==> {train_auc}")

        print("Predicting on main validation set ... ")
        val_pred = clf.predict_proba(val_feats)[:,1]
        val_auc = roc_auc_score(val_target, val_pred)
        print(f"==> {val_auc}")

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
    # from argparse import Namespace
    # args = Namespace(boosting_type='gbdt', cache_dir='/mnt/hdd/sbaio/asos/', cv=0, data_dir='data/', data_frac=1.0, heldout_frac=0.1, learning_rate=0.1, max_depth=3, model='light_gbdt', n_estimators=10, nfolds=5, num_leaves=150, outdir='out/main/', seed=7, task='main', test=1, use_customer_data=1, use_product_data=1, use_purchase_data=1, use_views_data=1, verbose=0)
    args.outdir = f"out/{args.task}/"
    if args.task == 'main':
        main(args)
    elif args.task == 'grid_search':
        param_ranges = {
            "learning_rate":[0.1,],#  0.2, 0.05
            "num_leaves":[150], #31, 50, 100, 
            "n_estimators":[100], # 200, 300
            "data_frac":[0.1, 0.5, 1]
        }
        grid_search(args, param_ranges)
    elif args.task == "data_ablation":
        param_ranges = {
            "n_estimators":[100],
            "num_leaves":[150,],
            "use_views_data":[1, 0],
            "use_product_data":[1, 0],
            "use_purchase_data":[1, 0],
            "use_customer_data":[1, 0],
        }
        grid_search(args, param_ranges)
    elif args.task == "data_size":
        param_ranges = {
            "n_estimators":[100],
            "data_frac":[0.1, 0.5, 1.0],
            "num_leaves":[100,], 
        }
        grid_search(args, param_ranges)

