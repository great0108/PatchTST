from data_provider.data_cluster import cluster_result
from multiprocessing import freeze_support
import numpy as np

if __name__=="__main__":
    freeze_support()
    cluster = cluster_result("Electricity", 100)
    counts = np.unique(cluster, return_counts=True)[1]
    print(counts)