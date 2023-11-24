import numpy as np
import torch

# instance norm - done
# mae loss - done
# various aug - done
# flatten head or first project -> channel mix
# add std input - done

a = np.array([[1,2], [1,2]])
print(np.stack(
        [np.arange(4)] * 4,
        axis=1).T)