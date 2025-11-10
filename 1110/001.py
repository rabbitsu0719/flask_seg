# HSV를 활용한 파란색 객체 분할
import cv2
import numpy as np

# 1. 이미지 로드
image = cv2.imread('image.png')

# 2. BGR 이미지를 HSV로 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 파란색 범위 정의 (HSV)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# 4. HSV 이미지에서 파란색 범위에 해당하는 픽셀만 추출하여 마스크 생성
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# 5. 원본 이미지에 마스크 적용 (파란색 영역만 남김)
result = cv2.bitwise_and(image, image, mask=mask)

mask_inv = cv2.bitwise_not(mask) # 반전


result2 = cv2.bitwise_and(image, image, mask=mask_inv)

#5-1. 알파 채널 추가 (투명도 적용)
rgba_image = cv2.cvtColor(result2, cv2.COLOR_BGR2BGRA)
# 반전된 마스크를 알파 채널로 사용
rgba_image[:, :, 3] = mask_inv
# 6. 결과 이미지 표시
cv2.imshow('Original Image', image)
cv2.imshow('Blue Mask', mask)
cv2.imshow('Segmented Blue Object', result)
cv2.imshow('Segmented Non-Blue Object', result2)
cv2.imwrite('image.png', rgba_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

