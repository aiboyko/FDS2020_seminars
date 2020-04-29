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



from sklearn import datasets, linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

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
    parser.add_argument('--flights_url', default="https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz", \
                        type=str, help="URL to download Flights data")
    parser.add_argument('--data_dir', default='data', type=str, help='root of the dataset')
    parser.add_argument('--n_rows', default=10000, type=int, help='number of rows extracted to jsons')
    parser.add_argument('--dask', default=False, type=bool, help='use dask')
    
    return parser
    
    
def main(args):
    download_flights(data_dir=args.data_dir, flights_targz_url=args.flights_url)
    untar(data_dir=args.data_dir)
    jsonize(data_dir=args.data_dir, n_rows=args.n_rows)
    
    dfs = load_dfs_from_jsons(useDask=args.dask)
    X, y = prepareData(dfs=dfs)
    
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    regr = linear_model.Lasso(alpha=0.5, normalize=0)
#     regr.fit(X_train, y_train)

    cluster = make_cluster()
    client = Client(cluster)

    param_grid = {
        'alpha': [1e-3, 1e-2, .5, 1],
        'normalize': [True, False],
        'max_iter': [10000]
    }

    grid_search = GridSearchCV(regr, param_grid, verbose=2, cv=2, n_jobs=-1)

    with joblib.parallel_backend("dask"):
        grid_search.fit(X_train, y_train)

    resultCV = grid_search2pd(grid_search)
    resultCV.to_csv('with_dask.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)