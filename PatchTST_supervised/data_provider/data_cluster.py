import pickle
import numpy as np
import torch
from layers.kShape import KShapeClusteringCPU
from multiprocessing import freeze_support
from data_provider.data_loader import Dataset_ETT_hour

def cluster_result(dataset, cluster_num):
    with open("cluster_result.pkl", "rb") as tf:
        cluster_dict = pickle.load(tf)

    if dataset in cluster_dict and cluster_num in cluster_dict[dataset]:
        return torch.tensor(cluster_dict[dataset][cluster_num])
    
    cluster = cluster_data(cluster_num)
    if not dataset in cluster_dict:
        cluster_dict[dataset] = {}
    cluster_dict[dataset][cluster_num] = cluster

    with open("cluster_result.pkl", "wb") as tf:
        pickle.dump(cluster_dict, tf)

    return torch.tensor(cluster)
    
def cluster_data(num_clusters):
    dataset = Dataset_ETT_hour("./data/ETT/", features="M")
    print(dataset.data_x.shape)

    univariate_ts_datasets = np.expand_dims(dataset.data_x.transpose(1, 0), axis=2)
    print(univariate_ts_datasets.shape)

    # CPU Model
    ksc = KShapeClusteringCPU(num_clusters, centroid_init='zero', max_iter=100, n_jobs=-1)
    ksc.fit(univariate_ts_datasets)

    labels = ksc.labels_ # or ksc.predict(univariate_ts_datasets)
    # cluster_centroids = ksc.centroids_

    print(labels)
    return labels

if __name__=="__main__":
    freeze_support()
    cluster_result("ETTh1", 4)
    