#이미 나온 결과가지고 실험하는 코드

import argparse

#Perform the ensemble
from dev_ensemble import *
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import os
import cv2

# data_dir = "./data/"

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, required = True, help='Directory where data is stored')
# parser.add_argument('--ensemble', action='store_true', help='Enable ensemble mode')
args = parser.parse_args()

data_dir = args.data_directory

# dataset_name: 'val' 'blurred_val' 'brightened_val' 'darkened_val'
predict_model = {"Kaggle_vgg11":'blurred_val', "Kaggle_googlenet": 'val',
                     "Kaggle_squeezenet": 'brightened_val', "Kaggle_wideresnet": 'darkened_val'}

model_name = "Kaggle_wideresnet"
dataset_name = "val"
prob1, labels = getfile(model_name + "_" + dataset_name + "_Model", root=data_dir)
prob1_Lap = get_Lap_file(model_name + "_" + dataset_name + "_Lap", root=data_dir)

model_name = "Kaggle_wideresnet"
dataset_name = "blurred_val"
prob2, _ = getfile(model_name + "_" + dataset_name + "_Model", root=data_dir)
prob2_Lap = get_Lap_file(model_name + "_" + dataset_name + "_Lap", root=data_dir)

model_name = "Kaggle_wideresnet"
dataset_name = "blurred_val"
prob3, _ = getfile(model_name + "_" + dataset_name + "_Model", root=data_dir)
prob3_Lap = get_Lap_file(model_name + "_" + dataset_name + "_Lap", root=data_dir)

model_name = "Kaggle_wideresnet"
dataset_name = "blurred_val"
prob4, _ = getfile(model_name + "_" + dataset_name + "_Model", root=data_dir)
prob4_Lap = get_Lap_file(model_name + "_" + dataset_name + "_Lap", root=data_dir)

ensemble_sugeno(labels,prob1,prob2,prob3,prob4, prob1_Lap, prob2_Lap, prob3_Lap, prob4_Lap)

