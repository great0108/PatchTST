import torch
import numpy as np

def distance_normal_aug(data, std=0.01, end=None):
    if end == None:
        end = len(data)

    noise = np.random.normal(0, np.repeat(
        np.expand_dims(np.arange(std, 0, -std/end), 0),
        data.shape[1], axis=0))
    noise = np.concatenate([noise, np.zeros((len(data) - end, data.shape[1]))])
    return data + noise

def normal_aug(data, std=0.01):
    noise = np.random.normal(0, std, size=data.shape)
    return data + noise

def slope_aug(data, slope=0.02):
    slope = np.repeat(
        np.expand_dims(np.arange(0, slope, slope/len(data)), 0),
        data.shape[1], axis=0)
    return data + slope