import os
import numpy as np
import json
import pickle as pkl

from PIL import Image
from glob import glob

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