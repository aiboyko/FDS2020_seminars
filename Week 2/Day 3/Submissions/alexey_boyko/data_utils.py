from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
import argparse
import json
from glob import glob

def download_flights(
    data_dir='data',
    flights_targz_url="https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz"):
    '''loading flights data from url, untaring, making jsons'''
    
    print("Setting up data directory")
    print("-------------------------")
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    flights_raw = os.path.join(data_dir, 'nycflights.tar.gz')
    if not os.path.exists(flights_raw):
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        url = flights_targz_url
        urllib.request.urlretrieve(url, flights_raw)
        print("done", flush=True)           
    print("** Finished! **")
    
def untar(data_dir='data'):
    flightdir = os.path.join(data_dir, 'nycflights')
    if not os.path.exists(flightdir):
        print("- Extracting flight data... ", end='', flush=True)
        tar_path = os.path.join('data', 'nycflights.tar.gz')
        with tarfile.open(tar_path, mode='r:gz') as flights:
            flights.extractall('data/')
        print("done", flush=True)

        
def jsonize(data_dir='data', n_rows=10000):
    jsondir = os.path.join(data_dir, 'flightjson')
    if not os.path.exists(jsondir):
        print("- Creating json data... ", end='', flush=True)
        os.mkdir(jsondir)
        for path in glob(os.path.join('data', 'nycflights', '*.csv')):
            prefix = os.path.splitext(os.path.basename(path))[0]
            # Just take the first 10000 rows for the demo
            df = pd.read_csv(path).iloc[:n_rows]
            df.to_json(os.path.join('data', 'flightjson', prefix + '.json'),
                       orient='records', lines=True)
        print("done", flush=True)

        
def load_dfs_from_jsons(data_dir='data', useDask=False, n_json_files=100):
    paths = glob(os.path.join(data_dir, 'flightjson/*json'))
    
    if not useDask:
        dfs = [pd.read_json(path, lines=True) for path in paths[:n_json_files]]
        dfs = pd.concat(dfs)
    else:
        from dask import delayed
        import dask.dataframe as dd

        import dask.bag as db #os.path.join(data_dir, 'flightjson/*json'
        mybag = delayed(db.read_text(paths).map(json.loads))

        mybag.to_dataframe()

        dfs = [(dd.read_json(path, lines=True)) for path in paths[:n_json_files]]
        dfs = dd.concat(dfs)
    
    return dfs

def prepareData(dfs,
 activeTextFeatures=['UniqueCarrier', 'Origin', 'Dest'], features2drop=['CRSElapsedTime'],
 useDask=False):
    '''X - pandas dataframe or its dask version'''
    allTextFeatures = set(['UniqueCarrier', 'Origin', 'Dest', 'TailNum'])

    
    dfs.columns.drop(['TaxiIn', 'TaxiOut'])
    dfs = dfs.dropna()
    y = dfs['DepDelay']
    
    dfs = dfs[dfs.columns.drop('DepDelay')]
    activeNonTextFeatures = dfs.columns.drop(allTextFeatures)
    activeNonTextFeatures = activeNonTextFeatures.drop(features2drop) #, 'ArrDelay'])    
    

    if not useDask:
        X = pd.concat([pd.get_dummies(dfs[col]) for col in activeTextFeatures], axis=1)  
        X = pd.concat([X, dfs[activeNonTextFeatures]], axis=1)
        X = X.dropna(axis=1)
    else:
        import dask.dataframe as dd
        # failed to implement categorical on delayed data
        # dfs.categorize(columns=activeTextFeatures)
        # X = dd.concat([dd.get_dummies(dfs[col]) for col in activeTextFeatures], axis=1)  
        # X = dd.concat([X, dfs[activeNonTextFeatures]], axis=1)
        X = dfs[activeNonTextFeatures]

    return X, y

def grid_search2pd(grid_search):
    other_params = list(grid_search.cv_results_.keys())
    other_params.remove('params')
    resultCV = pd.concat([pd.DataFrame({key: grid_search.cv_results_[key]}) for key in other_params ], axis=1)
    resultCV = pd.concat([pd.DataFrame(grid_search.cv_results_['params']), resultCV], axis=1)
    return resultCV