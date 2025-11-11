# 002.py — Final (Ultralytics SAMPredictor + Mobile SAM + point prompt)
from ultralytics.models.sam import Predictor as SAMPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# 1) 이미지 로드(RGB)
image = cv2.imread("example.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# (선택) 긴 변 768으로 축소하면 메모리 안정적
h, w = image_rgb.shape[:2]
scale = 768 / max(h, w)
if scale < 1.0:
    image_rgb = cv2.resize(image_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

# 2) 디바이스 (MPS 우선)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 3) 경량 SAM 사용(자동 다운로드). OOM 방지를 위해 imgsz=512 권장
WEIGHTS = "mobile_sam.pt"
overrides = dict(
    conf=0.25, task="segment", mode="predict",
    imgsz=512,              # 메모리/속도 안정
    model=WEIGHTS,
    device=device,
    save=False              # 자동 저장 끔
)
predictor = SAMPredictor(overrides=overrides)

# 4) 이미지 설정
predictor.set_image(image_rgb)

# 5) 포인트 프롬프트 (예시 좌표/라벨)
input_points = np.array([[300, 400]], dtype=np.float32)  # (N, 2)
input_labels = np.array([1], dtype=np.int32)             # (N,)  1=foreground

# 6) 호출형으로 추론 (predict() 메서드가 아님!)
results = predictor(points=input_points, labels=input_labels, multimask_output=False)

# 7) 결과에서 마스크 꺼내기
result = results[0]
masks_t = result.masks.data                 # torch.Tensor [N, Hm, Wm]
mask = (masks_t[0].detach().cpu().numpy() > 0.5).astype("uint8")

# 8) 마스크 크기를 현재 이미지 크기에 맞춤
H, W = image_rgb.shape[:2]
if mask.shape != (H, W):
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

# 9) 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.imshow(mask, alpha=0.6)
plt.axis('off')
plt.title("Ultralytics SAM (mobile) — point prompt")
plt.show()

# 10) 저장
cv2.imwrite("mask_result.png", mask * 255)
