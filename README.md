# polycentric-mobility

This code implements the algorithm described in _From centre to centres: polycentric structures in individual mobility_.

It is built using Python 3.7.7 on the Ubuntu 16.04.6 LTS OS. Package dependencies:
numpy 1.19.1, pandas 1.0.5, geopandas 0.8.1, scikit-learn 0.23.2, scipy 1.5.0, haversine 2.1, and (optionally for parallel processing) joblib 1.0.1.
All of these can be installed via pip or Anaconda. This code runs without any further software or hardware dependencies.

#### How to use

The code consists of a single file, main.py. In addition to utility functions and the trimmed k-means class, the file contains the `main` function. This function takes as input `datapath`, which is the path to the input pandas dataframe, and optionally `results_path` (defaults to `"results.pkl"`), `max_k` (`6`), and `trimming_coeff`(`0.9`). For the interpretation of `trimming_coeff`, we refer the user to the paper that this code accompanies.

The input data (stored as a pickled pandas dataframe) should have the following columns:
1. 'user': integer : ID for every user
2. 'loc': integer : ID for every location
3. 'lat': float : latitude of location
4. 'lon': float : longitude of location
5. 'start_time' and 'end_time' : pandas datetime _or_ 'time_spent' : pandas datetime or float
Note: the algorithm chooses the unweighted mean as the representative coordinates if the same location ID is associated with multiple coordinates.


The output is a pandas dataframe (also stored as `results_path`) with the columns:
1. 'user' : ID of user
2. 'com': centre of mass location
3. 'tcom': trimmed centre of mass location
4. 'rog': radius of gyration
5. 'L1': most visited location ID
6. 'L2': second most visited location ID
7. 'k': optimal k*
8. 'centers': locations of centres
9. 'auc_com': cover score of monocentric description from com
10. 'auc_1': cover score of monocentric description from trimmed com
11. 'auc_2': cover score of polycentric description with k=2 and centers found via t-k-means
12. 'auc_k': cover score of polycentric description with k=k* and centers found via t-k-means
13. 'auc_kmeans': cover score of polycentric description with k=k* and centers found via k-means


#### Demo

The repo contains `demo_data.pkl`, which is a synthetic dataset generated to demonstrate the working of the code. There are 300 unique locations in the dataset, sampled 100 each from three 2-dimensional Gaussians. The dataset contains records of three users (User 1, 2 and 3), with 600 records of equal weight per user. User _i_ has records sampled uniformly from _i_ clusters.

Running the `main` function on `demo_data.pkl` with default options creates `demo_results.pkl`. A single run takes 1min 46s measured via the `%%timeit` utility on Jupyter notebook (`5 loops, best of 5: 1min 46s per loop`).
