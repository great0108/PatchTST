import numpy as np
import torch

# instance norm
# mae loss
# flatten head -> channel mix
# add std input

a = np.array([[1,2], [1,2]])
print(np.stack(
        [np.arange(4)] * 4,
        axis=1).T)