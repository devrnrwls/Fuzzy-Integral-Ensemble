import cv2
import numpy as np

# 이미지를 로드합니다.
# image = cv2.imread('./food/blurred_train/Normal/haribo_1.png')
image = cv2.imread('data/box/normal.jpg')
image = cv2.resize(image, (1024, 768))

# 이미지를 그레이스케일로 변환합니다.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Laplacian을 계산합니다.
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Laplacian 분산을 계산합니다.
laplacian_var = laplacian.var()

# Laplacian 분산 값을 이미지에 표시합니다.
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, f'Laplacian Var: {laplacian_var:.2f}', (10, 90), font, 1, (0, 0, 255), 2)
# cv2.putText(image, f'Laplacian Var: 87.24', (10, 90), font, 1, (0, 0, 255), 2)

# 이미지를 화면에 표시합니다.
cv2.imshow('Image with Laplacian Var', image)

# 키 입력을 대기합니다.
cv2.waitKey(0)

# 창을 닫습니다.
cv2.destroyAllWindows()
