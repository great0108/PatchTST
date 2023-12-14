from data_provider.data_cluster import cluster_result
from multiprocessing import freeze_support

if __name__=="__main__":
    freeze_support()
    print(cluster_result("ETTh1", 7))