import torch
import numpy as np

def distance_normal_aug(data, std=0.01, end=None):
    if end == None:
        end = len(data)

    noise = np.random.normal(0, np.stack(
        [np.arange(std, 0, -std/end)] * data.shape[1],
        axis=1))
    print(np.stack(
        [np.arange(std, 0, -std/end)] * data.shape[1],
        axis=1))
    
    noise = np.concatenate([noise, np.zeros((len(data) - end, data.shape[1]))])
    return data + noise

def distance_std_aug(data, std=0.01, end=None):
    if end == None:
        end = len(data)
    noise = np.random.normal(0, np.arange(std, 0, -std/end).reshape(-1, 1) + np.zeros_like(data))
    noise = np.concatenate([noise, np.zeros((len(data) - end, data.shape[1]))])
    return data + noise

def normal_aug(data, std=0.01):
    noise = np.random.normal(0, std, size=data.shape)
    return data + noise

def slope_aug(data, slope=0.02):
    slope = np.stack(
        [np.arange(0, slope, slope/len(data))] * data.shape[1],
        axis=1)
    return data + slope


a = np.array([[1,1], [2,2], [3,3]])
distance_normal_aug(a, std=0.1)