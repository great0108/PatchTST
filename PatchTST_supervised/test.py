import numpy as np
import torch
from data_provider.data_loader import Dataset_ETT_hour
import matplotlib.pyplot as plt
from data_provider.data_cluster import cluster_data
from layers.KShape_gpu import KShapeClusteringGPU

        
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
# add feature_mix layernorm

n = 7
length = 500

dataset = Dataset_ETT_hour("./data/ETT/", features="M").data_x
dataset = dataset[1500 : 1500+length]
print(dataset.shape)
a = np.random.normal(scale=0.1, size=dataset.shape)
dataset += a
# univariate_ts_datasets = np.expand_dims(dataset.transpose(1, 0), axis=2)
# print(univariate_ts_datasets.shape)

# # Model
# ksc = KShapeClusteringGPU(4, centroid_init='zero', max_iter=20)
# ksc.fit(univariate_ts_datasets)

# labels = ksc.labels_ # or ksc.predict(univariate_ts_datasets)
# # cluster_centroids = ksc.centroids_

# print(labels)
# counts = np.unique(labels, return_counts=True)[1]
# print(counts)

plt.figure(figsize=(4, 5))
for i in range(n):
    y = dataset[:, i]
    ax = plt.subplot(n, 1, i+1)
    plt.subplots_adjust(hspace=0)
    plt.plot(y, color="g")
    plt.xticks(visible=False)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

plt.tight_layout()
plt.show()

