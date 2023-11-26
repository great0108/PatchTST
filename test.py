import numpy as np
import torch

# instance norm
# mae loss
# flatten head -> channel mix
# add std input
# seg transformer -> view range increase etc 2,4,8,...

a = np.array([[1,2], [1,2]])
print(np.stack(
        [np.arange(4)] * 4,
        axis=1).T)