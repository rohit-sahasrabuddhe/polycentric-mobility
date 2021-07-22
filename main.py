import numpy as np
import pandas as pd
import geopandas as gp
from sklearn.cluster import KMeans
from haversine import haversine_vector
from sklearn.metrics import auc
from scipy.spatial import distance_matrix as DM

from joblib import Parallel, delayed

# Functions for conversion from latlon to cartesian and back
def to_cartesian(lat, lon):
    lat, lon = np.pi * lat / 180, np.pi * lon / 180
    return np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
def to_latlon(x,y,z):
    lat, lon = np.arctan2(z, np.sqrt(x**2+y**2))*180/np.pi, np.arctan2(y, x)*180/np.pi
    return lat, lon

class TrimmedKMeans:
    def __init__(self, k, data, weights, cutoff):
        self.k = k
        self.data = data #A numpy array of size [N, 3]
        self.weights = weights / np.sum(weights) #size [N,]
        self.centers = self.data[np.random.choice(range(self.data.shape[0]), size=k, replace=False)]
        
        self.distance_matrix = DM(self.data, self.centers)
        self.cluster_assignment = np.argmin(self.distance_matrix, axis=1)
        self.distance = np.min(self.distance_matrix, axis=1)
        self.inertia = 0
        
        self.cutoff=cutoff
        
    def get_inertia_labels(self):
        self.distance_matrix = DM(self.data, self.centers)
        self.cluster_assignment = np.argmin(self.distance_matrix, axis=1)
        self.distance = np.min(self.distance_matrix, axis=1)
        self.inertia = 0
        for i in range(self.k): # Loop through all the clusters
            # get the coordinates, global weights and distance to center
            coords, weights, dists = self.data[self.cluster_assignment == i], self.weights[self.cluster_assignment == i], self.distance[self.cluster_assignment == i]
            if coords.shape[0] == 0:
                continue
            
            indices_asc = np.argsort(dists)
            coords, weights, dists = coords[indices_asc], weights[indices_asc], dists[indices_asc] # sort everything by the distance
            cluster_wt = np.sum(weights) # total weight of the cluster
            weights = weights / cluster_wt # this gives the local weight (within the cluster)
            weights_cumsum = np.cumsum(weights)
            
            last_entry = np.sum(weights_cumsum <= self.cutoff) + 1 # the index of the last location that needs to be looked at
            coords, weights, dists, weights_cumsum = coords[:last_entry].copy(), weights[:last_entry].copy(), dists[:last_entry].copy(), weights_cumsum[:last_entry].copy()
            # Remove the extra weight
            weights[-1] -= weights_cumsum[-1] - self.cutoff
            # Add to the inertia
            self.inertia += np.sum((weights * cluster_wt) * (dists**2))
        return np.sqrt(self.inertia), self.cluster_assignment
        
    def update(self):
        self.distance_matrix = DM(self.data, self.centers)
        self.cluster_assignment = np.argmin(self.distance_matrix, axis=1)
        self.distance = np.min(self.distance_matrix, axis=1)
        
        for i in range(self.k): # Loop through all the clusters
            # get the coordinates, global weights and distance to center
            coords, weights, dists = self.data[self.cluster_assignment == i], self.weights[self.cluster_assignment == i], self.distance[self.cluster_assignment == i]
            if coords.shape[0] == 0:
                continue
            
            indices_asc = np.argsort(dists)
            coords, weights, dists = coords[indices_asc], weights[indices_asc], dists[indices_asc] # sort everything by the distance
            cluster_wt = np.sum(weights) # total weight of the cluster
            weights = weights / cluster_wt # this gives the local weight (within the cluster)
            weights_cumsum = np.cumsum(weights)
            # last entry is the index of the last location that needs to be looked at
            last_entry = np.sum(weights_cumsum <= self.cutoff) + 1
            coords, weights, dists, weights_cumsum = coords[:last_entry].copy(), weights[:last_entry].copy(), dists[:last_entry].copy(), weights_cumsum[:last_entry].copy()
            # Remove the extra weight
            weights[-1] -= weights_cumsum[-1] - self.cutoff
            
            # Update the center
            weights = weights / np.sum(weights)
            self.centers[i] = np.average(coords, axis=0, weights=weights)       
        

    def plot(self):
        for i in range(self.k):
            plt.scatter(self.data[self.cluster_assignment == i][:, 0], self.data[self.cluster_assignment == i][:, 1])
        plt.scatter(self.centers[:, 0], self.centers[:, 1], marker='+', color='black', s=50)
    
    def get_best_fit(self):
        best_centers, best_inertia, best_labels = None , np.inf, None
        for _ in range(50): #compare across 50 random initializations
            c = np.inf
            self.centers = self.data[np.random.choice(range(self.data.shape[0]), size=self.k, replace=False)]
            for _ in range(50): #fixed number of iterations
                old_c = np.copy(self.centers)
                self.update()
                c = np.sum((self.centers - old_c)**2)
                if c == 0:
                    break
            this_inertia, this_labels = self.get_inertia_labels()
            if this_inertia < best_inertia:
                best_inertia = this_inertia
                best_labels = this_labels
                best_centers = self.centers
            if best_inertia == 0:
                break
            
        return best_centers, best_labels, best_inertia
    

def get_result(u, user_data, locs, max_k, trimming_coeff):
    #print(f"User {u}, {to_print}")
    result = {'user':u, 'com':None, 'tcom':None, 'rog':None, 'L1':None, 'L2':None, 'k':None, 'centers':None, 'auc_com':None, 'auc_1':None, 'auc_2':None, 'auc_k':None, 'auc_kmeans':None}
    def get_area_auc(x, k, max_area, df):
        centers = x
        dists = np.min(haversine_vector(list(df.coords), centers, comb=True), axis=0)
        df['distance'] = dists
        df['area'] = k * df['distance']**2
        df = df.sort_values('area')[['area', 'time_spent']]        
        df = df[df['area'] <= max_area]
        if df.empty:
            return 0        
        df.time_spent = df.time_spent.cumsum()        
        df['area'] = df['area'] / max_area
        x = [0] + list(df['area']) + [1]
        y = [0] + list(df.time_spent) + [list(df.time_spent)[-1]]
        return auc(x, y)
        
    user_data = user_data[['loc', 'time_spent']].groupby('loc').sum()
    try:
        user_data.time_spent = user_data.time_spent.dt.total_seconds()
    except:
        pass
    user_data.time_spent = user_data.time_spent / user_data.time_spent.sum()
    user_data['lat'] = locs.loc[user_data.index].lat
    user_data['lon'] = locs.loc[user_data.index].lon
    
    highest_gap = None
    best_auc = None
    best_gap = None
    best_k = 1
    best_centers = None    
    
    user_data['coords'] = list(zip(user_data.lat, user_data.lon))        
    user_data['x'], user_data['y'], user_data['z'] = to_cartesian(user_data['lat'], user_data['lon'])
    com = to_latlon(np.sum(user_data['x']*user_data.time_spent), np.sum(user_data['y']*user_data.time_spent), np.sum(user_data['z']*user_data.time_spent))
    dist = haversine_vector(list(user_data.coords), [com], comb=True)
    rog = np.sqrt(np.sum(user_data.time_spent.to_numpy() * (dist**2)))
    com_auc = get_area_auc(com, 1, rog**2, user_data.copy())    
    
    result['com'] = com
    result['rog'] = rog
    result['L1'], result['L2'] = list(user_data.sort_values('time_spent', ascending=False).coords[:2])
    result['auc_com'] = com_auc    
    
    train_data_list = []
    # find max min and shape outside loop
    lat_min, lat_max = user_data.lat.min(), user_data.lat.max()
    lon_min, lon_max = user_data.lon.min(), user_data.lon.max()
    size = user_data.shape[0]
    for i in range(50):
        train_data = user_data.copy()
        train_data['lat'] = np.random.uniform(low=lat_min, high=lat_max, size=size)
        train_data['lon'] = np.random.uniform(low=lon_min, high=lon_max, size=size)
        train_data['coords'] = list(zip(train_data.lat, train_data.lon))        
        train_data['x'], train_data['y'], train_data['z'] = to_cartesian(train_data['lat'], train_data['lon'])
            
        #find rog of this data
        com = to_latlon(np.sum(train_data['x']*train_data.time_spent), np.sum(train_data['y']*train_data.time_spent), np.sum(train_data['z']*train_data.time_spent))
        dist = haversine_vector(list(train_data.coords), [com], comb=True)
        train_rog = np.sqrt(np.sum(train_data.time_spent.to_numpy() * (dist**2)))   
        
        train_data_list.append((train_data, train_rog))
    
    
    for k in range(1, max_k+1):   
        Trim = TrimmedKMeans(k, user_data[['x','y', 'z']].to_numpy(), weights = user_data.time_spent.to_numpy(), cutoff=trimming_coeff)
        true_centers, _, _ = Trim.get_best_fit()        
        true_centers = np.array([np.array(to_latlon(*i)) for i in true_centers])
        true_auc = get_area_auc(true_centers, k, rog**2, user_data.copy())
        
        if k == 1:
            result['tcom'] = tuple(true_centers[0])
            result['auc_1'] = true_auc
        if k== 2:
            result['auc_2'] = true_auc
        
        new_aucs = []
        for train_data, train_rog in train_data_list:
            Trim = TrimmedKMeans(k, train_data[['x','y', 'z']].to_numpy(), weights = train_data.time_spent.to_numpy(), cutoff=trimming_coeff)
            centers, _, _ = Trim.get_best_fit()        
            centers = np.array([np.array(to_latlon(*i)) for i in centers])
            new_aucs.append(get_area_auc(centers, k, train_rog**2, train_data.copy()))
            
            
        new_mean = np.mean(new_aucs)
        new_std = np.std(new_aucs)        
        gap = true_auc - new_mean
        
        if k == 1:
            highest_gap = gap
            best_gap = gap
            best_auc = true_auc
            best_centers = true_centers
            best_k = 1
            continue
        
        
        if gap - new_std > highest_gap:
            best_auc = true_auc
            best_gap = gap
            best_centers = true_centers
            best_k = k
        highest_gap = max(highest_gap, gap)
  
    
    result['k'] = best_k
    result['auc_k'], result['centers'] = best_auc, list(best_centers)
    
    kmeans = KMeans(result['k'])
    kmeans.fit(user_data[['x','y', 'z']].to_numpy(), sample_weight = user_data.time_spent.to_numpy())
    kmeans_centers = np.array([np.array(to_latlon(*i)) for i in kmeans.cluster_centers_])
    result['auc_kmeans'] = get_area_auc(kmeans_centers, result['k'], rog**2, user_data.copy())
    return result

def main(data_path, results_path="results.pkl", max_k=6, trimming_coeff=0.9):
    data = pd.read_pickle(data_path)
    try:
        data['time_spent'] = data['end_time'] - data['start_time']
    except:
        pass
    user_list = sorted(data.user.unique())
    locs = data[['loc', 'lat', 'lon']].groupby('loc').mean().copy()
    
    result = pd.DataFrame(Parallel(n_jobs=-1)(delayed(get_result)(u, data[data.user == u], locs, max_k, trimming_coeff) for u in user_list)).set_index('user')
    result.to_pickle(results_path)
    return result
