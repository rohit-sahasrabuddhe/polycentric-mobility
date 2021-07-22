# polycentric-mobility

This code implements the algorithm described in _From centre to centres: polycentric structures in individual mobility_.

It is built using Python 3.7.7 on the Ubuntu 16.04.6 LTS OS. Package dependencies:
numpy 1.19.1, pandas 1.0.5, geopandas 0.8.1, scikit-learn 0.23.2, scipy 1.5.0, haversine 2.1, and (optionally for parallel processing) joblib 1.0.1.
All of these can be installed via pip or Anaconda. This code runs without any further software or hardware dependencies.

The code takes as input a pandas dataframe (the variable 'data') with the following columns
1. 'user': an int ID for every user
2. 'start_time', 'end_time'
3. 'loc': an int ID for every location
4. 'lat', 'lon': floats

The code saves the resulting dataframe into 'results.pkl', with the columns
1. 'user'
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

