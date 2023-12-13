import numpy as np
import torch
from layers.kShape import KShapeClusteringCPU
from multiprocessing import freeze_support
from data_provider.data_loader import Dataset_ETT_hour

# if __name__=="__main__":
#     freeze_support()

#     dataset = Dataset_ETT_hour("./data/ETT/", features="M")
#     print(dataset.data_x.shape)

#     univariate_ts_datasets = np.expand_dims(dataset.data_x.transpose(1, 0), axis=2)
#     print(univariate_ts_datasets.shape)
#     num_clusters = 3

#     # CPU Model
#     ksc = KShapeClusteringCPU(num_clusters, centroid_init='zero', max_iter=100, n_jobs=-1)
#     ksc.fit(univariate_ts_datasets)

#     labels = ksc.labels_ # or ksc.predict(univariate_ts_datasets)
#     cluster_centroids = ksc.centroids_

#     print(labels)
#     print(cluster_centroids.shape)
        
# instance norm - done
# mae loss - done
# various aug - done
# flatten head or first project layer -> channel mix - done
# add std input - done
# seg transformer -> view range increase etc 2,4,8,... - done
# reconstruction task
# without pos encoding
# add pos encoding each tf layer
# add feature_mix dim param

a = torch.tensor([2, 1, 2, 1, 0, 0, 1])
b = torch.sort(a)[0]
order = torch.sort(a).indices
order2 = torch.sort(order).indices
print(b[order2])