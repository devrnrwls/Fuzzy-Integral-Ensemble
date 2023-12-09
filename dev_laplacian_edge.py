import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread('/data/package/train/Normal/2023-09-14_22-02-26.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('/home/son/Work/Fuzzy-Integral-Covid-Detection/package/blurred_train/Normal/2023-09-14_22-02-20.jpg', cv2.IMREAD_GRAYSCALE)

# Laplace 연산자 적용
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# 부호 변경을 위해 절댓값 취하기
laplacian_abs = np.absolute(laplacian)

# 결과 이미지를 8-bit로 변환
laplacian_8bit = np.uint8(laplacian_abs)

# 원본 이미지와 Laplace 엣지 각각 표시
cv2.imshow('Original Image', image)
cv2.imshow('Laplace Edge', laplacian_8bit)

# 키보드 입력 대기 후 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
