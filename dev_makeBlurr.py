import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

import cv2
#covid
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

#metal
mean = np.array([0.7399833, 0.67764108, 0.64568031])
std = np.array([0.10772628, 0.17323852, 0.18402071])

# parser = argparse.ArgumentParser()
# parser.add_argument('--data_directory', type=str, required = True, help='Directory where data is stored')
# args = parser.parse_args()

data_dir = 'data/jelly/'

#'/COVID' '/non-COVID'
data_category = '/Abnormal'
data_category_non ='/Normal'

#train val
original = "train"

# blurred brightened darkened
category = "brightened"

blurr_dir = data_dir + '/'+category+'_' + original
os.makedirs(blurr_dir, exist_ok=True)  # 디렉터리가 이미 존재하면 무시

blurr_dir_COVID = blurr_dir + data_category
os.makedirs(blurr_dir_COVID, exist_ok=True)  # 디렉터리가 이미 존재하면 무시

blurr_dir_nonCOVID = blurr_dir + data_category_non
os.makedirs(blurr_dir_nonCOVID, exist_ok=True)  # 디렉터리가 이미 존재하면 무시

data_transforms = {
    original: transforms.Compose([
        # transforms.Resize((224,224)),
        # transforms.Resize((720,720)),
        transforms.Resize((900,900)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

image_datasets = {original: datasets.ImageFolder(os.path.join(data_dir, original),
                                                 data_transforms[original], )}

testloader = torch.utils.data.DataLoader(image_datasets[original], batch_size=1, shuffle=False)

for idx, data in enumerate(testloader):
    images, labels = data
    image_np = images.squeeze().numpy()
    output_filename = os.path.basename(testloader.dataset.imgs[idx][0])
    # 이미지 데이터를 0-255 범위로 스케일링
    image_np = (image_np * std[:, None, None]) + mean[:, None, None]
    image_np = (image_np * 255).astype(np.uint8)

    image_np = np.transpose(image_np, (1, 2, 0))

    opencv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # #brightness

    brightness = 50  # 밝게 만들기 예시, 음수 값으로 어둡게 만들 수 있습니다.
    blurred_image = None

    # #blurr
    if category =="blurred":
        blurred_image = cv2.GaussianBlur(opencv_image, (15, 15), 0)  # (15, 15)은 커널 크기, 0은 표준 편차입니다.

    # #brightened
    elif category =="brightened":
        blurred_image = np.where((255 - opencv_image) < brightness, 255, opencv_image + brightness)

    #darkened
    elif category == "darkened":
        blurred_image = np.where(opencv_image < brightness, 0, opencv_image - brightness)
    else:
        print("category check~")

    if opencv_image is not None:
        if labels == 0: #COVID
            output_path = os.path.join(blurr_dir_COVID, output_filename)
            cv2.imwrite(output_path, blurred_image)
        else: #nonCOVID
            output_path = os.path.join(blurr_dir_nonCOVID, output_filename)
            cv2.imwrite(output_path, blurred_image)

        print(f'이미지가 {output_path}에 저장되었습니다.')

