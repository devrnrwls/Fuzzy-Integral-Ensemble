import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

import cv2

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, required = True, help='Directory where data is stored')
args = parser.parse_args()

data_dir = args.data_directory

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

image_datasets = {'val': datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])}

testloader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1)

for idx, data in enumerate(testloader):
    images, labels = data
    image_np = images.squeeze().numpy()

    # 이미지 데이터를 0-255 범위로 스케일링
    image_np = (image_np * std[:, None, None]) + mean[:, None, None]
    image_np = (image_np * 255).astype(np.uint8)

    image_np = np.transpose(image_np, (1, 2, 0))

    grey = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grey, (15, 15), 0)  # (15, 15)은 커널 크기, 0은 표준 편차입니다.

    cv2.imshow('blurred_image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    var = cv2.Laplacian(blurred_image, cv2.CV_64F).var()
    # if(var <200):
    #     if idx<376:
    #         print(f"idx: {idx+1}, var: {var}")
    #     else:
    #         print(f"idx: {idx - 375}, var: {var}")

    print(var)


