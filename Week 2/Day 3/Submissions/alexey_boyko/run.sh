echo nodask
python main.py --n_rows=9999 --out_file nodask_results
echo dask
python main.py --n_rows=9999 --dask --out_file dask_results