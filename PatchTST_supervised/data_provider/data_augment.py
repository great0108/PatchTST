import torch
import numpy as np

def distance_std_aug(data, std=0.01, end=None):
    if end == None:
        end = len(data)
    noise = np.random.normal(0, np.arange(std, 0, -std/end).reshape(-1, 1) + np.zeros_like(data))
    noise = np.concatenate([noise, np.zeros((len(data) - end, data.shape[1]))])
    return data + noise
