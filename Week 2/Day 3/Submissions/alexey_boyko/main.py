from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
import argparse
import joblib
import argparse



from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score


from dask.distributed import Client
from glob import glob
from data_utils import download_flights, untar, jsonize, load_dfs_from_jsons, prepareData, grid_search2pd


def make_cluster(**kwargs):
    try:
        from dask_kubernetes import KubeCluster
        kwargs.setdefault('n_workers', 8)
        cluster = KubeCluster(**kwargs)
    except ImportError:
        from distributed.deploy.local import LocalCluster
        cluster = LocalCluster()
    return cluster


def set_args(parser):
    parser.add_argument('--dask', action='store_true', help='use dask')
    parser.add_argument('--flights_url', default="https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz", \
                        type=str, help="URL to download Flights data")
    parser.add_argument('--data_dir', default='data', type=str, help='root of the dataset')
    parser.add_argument('--n_rows', default=10000, type=int, help='number of rows extracted to jsons')
    parser.add_argument('--out_file', default=10000, type=str, help='name of file to save results')
    
    return parser


def main(args):
    # download and prepare data
    download_flights(data_dir=args.data_dir, flights_targz_url=args.flights_url)
    untar(data_dir=args.data_dir)
    jsonize(data_dir=args.data_dir, n_rows=args.n_rows)
    
    dfs = load_dfs_from_jsons(useDask=args.dask)
    print(dfs)
    X, y = prepareData(dfs=dfs, useDask=args.dask)
   
    
    if not args.dask:
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import GridSearchCV
        param_grid = {
        'alpha': [1e-3, 1e-2, .5, 1],
        'normalize': [True, False]
        }
        regr = linear_model.Lasso(max_iter=10000)
        grid_search = GridSearchCV(regr, param_grid, cv=2, n_jobs=-1, verbose=2)
    else:
        from dask_ml.xgboost import XGBRegressor
        from dask_ml.model_selection import train_test_split
        from dask_ml.model_selection import GridSearchCV
        param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [1e-2, 1e-1],
        'n_estimators': [10, 100],
        }
        regr = XGBRegressor()
        grid_search = GridSearchCV(regr, param_grid, cv=2, n_jobs=-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    cluster = make_cluster()
    client = Client(cluster)
    
    with joblib.parallel_backend("dask"):
        print('Doing gridsearch...')
        grid_search = GridSearchCV(regr, param_grid, cv=2, n_jobs=-1) #dask gridsearch does not support verbose
        grid_search.fit(X_train, y_train)

    resultCV = grid_search2pd(grid_search)
    resultCV.to_csv(args.out_file + '.csv')
    print('gridsearch finished')
    print('mean tests scores: ', resultCV['mean_test_score'])
    print('mean fit times: ', resultCV['mean_fit_time'])
    return


if __name__ == '__main__':
    #init parser
    parser = argparse.ArgumentParser()
    set_args(parser)
    
    #parse
    args = parser.parse_args()
    print('==============')
    print(args)
    print('==============')
    main(args)