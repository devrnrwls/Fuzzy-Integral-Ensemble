import numpy as np
import os
from PIL import Image

# 이미지 파일이 있는 디렉토리 경로 설정
# image_dir = './data/train/Normal'  # 이미지 파일들이 있는 디렉토리 경로로 변경해야 합니다.
# image_dir = './Covid/train/COVID'  # 이미지 파일들이 있는 디렉토리 경로로 변경해야 합니다.
image_dir = 'data/package/train/Normal'  # 이미지 파일들이 있는 디렉토리 경로로 변경해야 합니다.

# 이미지 파일 목록을 가져오는 함수
def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# 이미지 데이터의 평균과 표준편차 계산 함수
def calculate_mean_and_std(image_paths):
    # 이미지 데이터를 저장할 리스트 초기화
    image_data = []

    # 이미지 파일을 열어서 NumPy 배열로 변환하여 리스트에 추가
    for image_path in image_paths:
        image = Image.open(image_path)
        #이미지 사이즈가 다르면 에러 발생
        # image = image.convert("RGB")
        # new_size = (224, 224)
        # image = image.resize(new_size)
        image = np.array(image)
        image_data.append(image)

    # 이미지 데이터를 NumPy 배열로 변환
    image_data = np.array(image_data)

    # 이미지 데이터의 평균과 표준편차 계산
    mean = np.mean(image_data, axis=(0, 1, 2))
    std = np.std(image_data, axis=(0, 1, 2))

    return mean, std

# 이미지 파일 목록 가져오기
image_paths = get_image_paths(image_dir)

# 평균과 표준편차 계산(나누기 255를 해줘야 소수점 값이 나옴)
mean, std = calculate_mean_and_std(image_paths)
mean = mean/255
std = std/255

print(f'Mean: {mean}')
print(f'Std: {std}')
