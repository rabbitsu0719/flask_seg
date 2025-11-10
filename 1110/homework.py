import cv2
import numpy as np

# 1) 이미지 불러오기
img = cv2.imread("face.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2) 피부색 범위 설정 (HSV 기반)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([0, 30, 60], dtype=np.uint8)
upper = np.array([20, 150, 255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower, upper)

# 3) 마스크를 이용해 알파 채널 생성 (피부 영역만 남기고 나머지는 투명)
b, g, r = cv2.split(img)
alpha = mask

rgba = cv2.merge([b, g, r, alpha])

# 4) PNG로 저장
cv2.imwrite("result.png", rgba)

print("✅ result.png 생성 완료")
