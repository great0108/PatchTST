import numpy as np
import torch

# instance norm
# mae loss
# flatten head -> channel mix
# add std input

a = np.repeat(np.expand_dims(np.arange(4), 0), 4, axis=0)
print(a)
print(np.expand_dims(np.arange(4), 0))