from ultralytics import FastSAM
import cv2
import matplotlib.pyplot as plt

# SAM 모델 로드
model = FastSAM("sam_b.pt") 

# 이미지 경로
image_path = "example.jpg"

# 전체 이미지 분할 (자동)
results = model(image_path)

# 결과 마스크 추출
masks = results[0].masks.data.cpu().numpy()  # 여러 마스크가 있을 수 있음

# 첫 번째 마스크만 선택
mask = masks[0]

# 원본 이미지 로드
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = (mask > 0.5).astype("uint8")
mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
segmented = image_rgb.copy()
segmented[mask == 0] = [0, 0, 0]
# 마스크 시각화
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Segmentation Mask")
plt.imshow(mask, cmap='gray')
plt.axis('off')

# 분할 결과 이미지 (마스크 적용)
segmented = image_rgb.copy()
segmented[mask == 0] = [0, 0, 0]  # 마스크가 0인 부분을 검정으로
plt.subplot(1, 3, 3)
plt.title("Segmented Object")
plt.imshow(segmented)
plt.axis('off')


plt.show()

# 마스크 및 분할 결과 저장
cv2.imwrite("mask_result.png", mask * 255)  # 마스크 저장
cv2.imwrite("segmented_result.png", cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))