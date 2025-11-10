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

# 3) Skin Mask (흰색 = 피부)
skin_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# 4) Face Removed (피부 영역만 제거)
mask_inv = cv2.bitwise_not(mask)
face_removed = cv2.bitwise_and(img, img, mask=mask_inv)

# 5) Transparent 출력
b, g, r = cv2.split(img)
alpha = mask_inv  # 피부는 투명(0), 나머지 불투명
rgba = cv2.merge([b, g, r, alpha])
cv2.imwrite("result.png", rgba)

# 6) 창으로 보기
cv2.imshow("Original Image", img)
cv2.imshow("Skin Mask", skin_mask)
cv2.imshow("Face Removed (using HSV)", face_removed)

cv2.waitKey(0)
cv2.destroyAllWindows()
