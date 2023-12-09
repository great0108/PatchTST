import numpy as np
import torch

# instance norm - done
# mae loss - done
# various aug - done
# flatten head or first project layer -> channel mix - done
# add std input - done
# seg transformer -> view range increase etc 2,4,8,... - done
# reconstruction task
# without pos encoding
# cluster time series
# mlp mixer with orthgonal
# simsiam?

a = np.array([[1,2], [1,2]])
print(np.stack(
        [np.arange(4)] * 4,
        axis=1).T)