# 005_sam_fridge_floor_meta.py  (Meta 공식 SAM 사용)
# pip install git+https://github.com/facebookresearch/segment-anything.git
# pip install opencv-python matplotlib requests

from segment_anything import sam_model_registry, SamPredictor
import cv2, numpy as np, matplotlib.pyplot as plt, requests, os, torch

IMG_PATH = "example.jpg"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CKPT = "sam_vit_b_01ec64.pth"          # SAM ViT-B 가중치

# 0) 체크포인트 준비(없으면 다운로드)
if not os.path.exists(CKPT):
    print(">> downloading SAM ViT-B weights...")
    with open(CKPT, "wb") as f:
        f.write(requests.get(CKPT_URL).content)

# 1) 이미지 로드 (+ RGB)
img_bgr = cv2.imread(IMG_PATH)
assert img_bgr is not None, f"이미지 없음: {IMG_PATH}"
img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H0, W0 = img_rgb_orig.shape[:2]

# 2) 메모리 안전 리사이즈 (긴 변 1024)
max_side = 1024
scale = min(1.0, max_side / max(H0, W0))
if scale < 1.0:
    W, H = int(W0 * scale), int(H0 * scale)
    img_rgb = cv2.resize(img_rgb_orig, (W, H), interpolation=cv2.INTER_LINEAR)
else:
    img_rgb, H, W = img_rgb_orig.copy(), H0, W0

# 3) SAM 로드(+MPS/CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=CKPT).to(device)
predictor = SamPredictor(sam)
predictor.set_image(img_rgb)

# ---------------- A) 냉장고(박스 프롬프트) ----------------
# 오른쪽 위 직사각형(냉장고 영역) — 비율로 지정해 이미지 크기 변화에 견고
bx1, by1 = int(W*0.58), int(H*0.04)
bx2, by2 = int(W*0.95), int(H*0.62)
box = np.array([[bx1, by1, bx2, by2]], dtype=np.float32)      # shape (1,4)

masks_box, scores_box, logits_box = predictor.predict(
    point_coords=None, point_labels=None,
    boxes=box,
    multimask_output=False
)
mask_box = masks_box[0].astype(np.uint8)  # (H, W)

# ---------------- B) 바닥(점 프롬프트) ----------------
# 바닥 여러 지점에 포그라운드(1) 점 몇 개
pts = np.array([
    [int(W*0.40), int(H*0.88)],
    [int(W*0.60), int(H*0.90)],
    [int(W*0.80), int(H*0.82)],
    [int(W*0.52), int(H*0.78)],
], dtype=np.float32)
lbl = np.ones(len(pts), dtype=np.int32)

masks_pts, scores_pts, logits_pts = predictor.predict(
    point_coords=pts, point_labels=lbl,
    boxes=None,
    multimask_output=False
)
mask_pts = masks_pts[0].astype(np.uint8)

# 4) (선택) 원본 크기 복원
if (H, W) != (H0, W0):
    mask_box = cv2.resize(mask_box, (W0, H0), interpolation=cv2.INTER_NEAREST)
    mask_pts = cv2.resize(mask_pts, (W0, H0), interpolation=cv2.INTER_NEAREST)
    img_show = img_rgb_orig
    # 점 좌표도 원본 좌표로 스케일 복원
    pts_show = (pts / scale).astype(np.float32)
else:
    img_show = img_rgb
    pts_show = pts

# 5) 제출용 좌/우 Figure
plt.figure(figsize=(12, 6))

# 왼쪽: 냉장고 (Box)
plt.subplot(1, 2, 1)
plt.imshow(img_show)
plt.imshow(mask_box, alpha=0.6, cmap="viridis")
# 박스도 시각적으로 표시
x1, y1, x2, y2 = int(bx1/scale), int(by1/scale), int(bx2/scale), int(by2/scale)
plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c="yellow", linewidth=2)
plt.title("Ultralytics SAM result (Box Prompt)")
plt.axis("off")

# 오른쪽: 바닥 (Points)
plt.subplot(1, 2, 2)
plt.imshow(img_show)
plt.imshow(mask_pts, alpha=0.6, cmap="viridis")
plt.scatter(pts_show[:,0], pts_show[:,1], c="yellow", s=80, edgecolors="black")
plt.title("Ultralytics SAM result (Point Prompt)")
plt.axis("off")

plt.tight_layout()
plt.show()

# 6) 저장물(캡처 대신 파일 제출 원하면)
cv2.imwrite("mask_fridge.png", (mask_box*255))
cv2.imwrite("mask_floor.png",  (mask_pts*255))
