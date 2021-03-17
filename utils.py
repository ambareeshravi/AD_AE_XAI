import os
import numpy as np
import json
import pickle as pkl
import matplotlib.pyplot as plt

from PIL import Image
from glob import glob
from tqdm import tqdm

from c2d_models import *

def load_json(file):
    if ".json" not in file: file += ".json"
    with open(file, "r") as f:
        contents = json.load(f)
    return contents

def dump_json(contents, file):
    if ".json" not in file: file += ".json"
    with open(file, "w") as f:
        json.dump(contents, f)
    return True

def load_pickle(file):
    if ".pkl" not in file: file += ".pkl"
    with open(file, "rb") as f:
        contents = pkl.load(f)
    return contents
    
def dump_pickle(contents, file):
    if ".pkl" not in file: file += ".pkl"
    with open(file, "wb") as f:
        pkl.dump(contents, f)
    return True

def read_image(image_path, resize_to = None):
    img = Image.open(image_path)
    if resize_to != None:
        img = img.resize(resize_to)
    return np.array(img)

def save_image(image_array, file_path):
    try:
        image_array = im_to_255(image_array)
        Image.fromarray(image_array).save(file_path)
        return True
    except Exception as e:
        print(e)
        return False

def join_paths(paths):
    path = ""
    for tag in paths:
        path = os.path.join(path, tag)
    return path

def read_directory_contents(directory):
    if "*" not in directory: directory = join_paths([directory, "*"])
    return sorted(glob(directory))

def create_directory(path):
    if not os.path.exists(path): os.mkdir(path)
        
def INFO(*list_of_strings):
    list_of_strings = list(list_of_strings)
    print("-"*40)
    print("\n".join(list_of_strings))
    print("-"*40)
    
def normalize(x):
    return (x - x.min())/(x.max() - x.min())

def im_to_255(x):
    if x.max() <= 1: return (x*255).astype(np.uint8)
    return x

def get_model(model_path, rec = True, max_value=1000):
    if rec: model = C2D_AE_128_3x3(isTrain = True)
    else: model = C2D_AE_128_3x3(isTrain = False, max_value = max_value)
    model.model.load_weights(model_path)
    return model.model

def im_3(x, channel_axis = -1):
    if len(x.shape) < 3:
        x = np.expand_dims(x, axis = channel_axis)
    if x.shape[channel_axis] < 3:
        x = x.repeat((1 + 3 - x.shape[channel_axis]), axis = channel_axis)
    return x
        