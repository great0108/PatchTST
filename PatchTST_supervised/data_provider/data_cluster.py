import pickle
import numpy as np
import torch
from layers.kShape import KShapeClusteringCPU
from layers.KShape_gpu import KShapeClusteringGPU
from multiprocessing import freeze_support
from data_provider.data_loader import *

def cluster_result(dataset, cluster_num, cluster_size=None):
    with open("cluster_result.pkl", "rb") as tf:
        cluster_dict = pickle.load(tf)

    if dataset in cluster_dict and cluster_num in cluster_dict[dataset]:
        if cluster_size:
            return torch.tensor(split_cluster(cluster_dict[dataset][cluster_num], cluster_size))
        return torch.tensor(cluster_dict[dataset][cluster_num])
    
    cluster = cluster_data(cluster_num)
    if not dataset in cluster_dict:
        cluster_dict[dataset] = {}
    cluster_dict[dataset][cluster_num] = cluster

    with open("cluster_result.pkl", "wb") as tf:
        pickle.dump(cluster_dict, tf)

    if cluster_size:
        cluster = split_cluster(cluster, cluster_size)

    return torch.tensor(cluster)
    
def cluster_data(num_clusters):
    # dataset = Dataset_ETT_hour("./data/ETTh1/", features="M")
    dataset = Dataset_Custom("./data/Electricity/", features="M", data_path="electricity.csv")
    print(dataset.data_x.shape)

    univariate_ts_datasets = np.expand_dims(dataset.data_x.transpose(1, 0), axis=2)
    print(univariate_ts_datasets.shape)

    # Model
    ksc = KShapeClusteringGPU(num_clusters, centroid_init='zero', max_iter=20)
    ksc.fit(univariate_ts_datasets)

    labels = ksc.labels_ # or ksc.predict(univariate_ts_datasets)
    # cluster_centroids = ksc.centroids_

    print(labels)
    counts = np.unique(labels, return_counts=True)[1]
    print(counts)
    return labels

def split_cluster(labels, cluster_size):
    index = np.argsort(labels)
    labels = np.sort(labels)
    result = np.zeros(len(labels))
    num = 0
    count = 0
    last = labels[0]
    for i in range(len(labels)):
        if last != labels[i]:
            last = labels[i]
            num += 1
            count = 0

        if count == cluster_size:
            count = 0
            num += 1

        count += 1
        result[index[i]] = num
    return result

    