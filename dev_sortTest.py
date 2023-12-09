import os
from torchvision import datasets, transforms
import numpy as np
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

import cv2

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_dir = './data'  # 데이터 디렉터리 경로를 지정하세요.
subfolders = ['COVID', 'non-COVID']

# 데이터 변환 설정
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# 정렬된 파일 리스트를 사용하여 ImageFolder 데이터셋 생성
image_datasets = {}
for phase in ['val']:
    sorted_file_lists = {}
    for subfolder in subfolders:
        folder_path = os.path.join(data_dir, phase, subfolder)
        file_list = sorted(os.listdir(folder_path))
        sorted_file_lists[subfolder] = file_list

    image_datasets[phase] = datasets.ImageFolder(
        os.path.join(data_dir, phase),
        data_transforms[phase],
        loader=lambda x: os.path.join(phase, x),  # 파일 경로를 조합하여 정렬된 파일 리스트 사용
    )

testloader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=False)

for idx, data in enumerate(testloader):
    images, labels = data
