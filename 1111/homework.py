# auto_pick_chair_and_nose.py
# pip install git+https://github.com/facebookresearch/segment-anything.git
# pip install opencv-python matplotlib requests

import os, requests, cv2, numpy as np, matplotlib.pyplot as plt, torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

IMG_PATH = "example.jpg"
CKPT = "sam_vit_b_01ec64.pth"
URL  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# 0) 체크포인트
if not os.path.exists(CKPT):
    print(">> downloading SAM checkpoint...")
    with open(CKPT, "wb") as f:
        f.write(requests.get(URL).content)

# 1) 이미지
img_bgr = cv2.imread(IMG_PATH); assert img_bgr is not None, "example.jpg 없음"
img_rgb0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H0, W0 = img_rgb0.shape[:2]

# 메모리 안전 리사이즈(긴 변 1024)
scale = min(1.0, 1024 / max(H0, W0))
img_rgb = cv2.resize(img_rgb0, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_LINEAR) if scale < 1.0 else img_rgb0.copy()
H, W = img_rgb.shape[:2]

# 2) SAM + 자동마스크(예: CPU 강제)
sam = sam_model_registry["vit_b"](checkpoint=CKPT).to("cpu")
gen = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=24,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    box_nms_thresh=0.7
)
masks = gen.generate(img_rgb)   # list of dicts with "segmentation","area","bbox",...

# 공용 유틸
def to_orig(mask_bool):
    m = mask_bool.astype(np.uint8)
    return cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST) if (H,W)!=(H0,W0) else m

def mask_stats(mask_bool, img_rgb):
    """면적, bbox, 중심, 원형도, 평균밝기(V)"""
    seg = mask_bool.astype(np.uint8)
    area = int(seg.sum())
    if area == 0: 
        return None
    yx = np.argwhere(seg>0)
    y1,x1 = yx.min(0); y2,x2 = yx.max(0)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    # 원형도 = 4πA / P^2
    cnts,_ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = cv2.arcLength(cnts[0], True) if cnts else 1.0
    circ = float(4*np.pi*area / (peri*peri + 1e-6))
    # 밝기(HSV V 평균)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    V = hsv[:,:,2]
    mean_v = float(V[seg==1].mean()) if area>0 else 255.0
    return dict(area=area, bbox=(x1,y1,x2-x1,y2-y1), cx=cx, cy=cy, circ=circ, mean_v=mean_v)

# 3) 의자 선택(왼쪽 아래, 중간크기, 직사각형)
def pick_chair(masks, img_rgb):
    img_area = H * W
    best = None
    best_score = -1

    for m in masks:
        st = mask_stats(m["segmentation"], img_rgb)
        if not st:
            continue

        x, y, w, h = st["bbox"]

        # ✅ 의자 위치: 아래쪽 25%, 좌측 25~55%
        if not (W*0.25 <= st["cx"] <= W*0.55):
            continue
        if st["cy"] < H*0.75:
            continue

        # ✅ 의자 크기: 0.3% ~ 8% 영역
        if not (0.003*img_area <= st["area"] <= 0.08*img_area):
            continue

        # ✅ 의자 모양: 가로가 세로보다 넓음
        ratio = w / max(h, 1)
        if not (1.2 <= ratio <= 3.0):
            continue

        # ✅ 책상보다 약간 더 어두움
        if st["mean_v"] > 160:
            continue

        # ✅ 점수: 더 넓고, 더 어둡고, 더 가로로 긴 것 선호
        score = st["area"] * ratio * (180 - st["mean_v"])
        if score > best_score:
            best_score = score
            best = m

    return best


# 4) 강아지 코 선택(우측 중앙, 작고 어두움, 원형도↑)
def pick_dog_nose(masks, img_rgb):
    img_area = H*W
    best = None; best_rank = (999, )  # (밝기, -원형도, 면적)
    for m in masks:
        st = mask_stats(m["segmentation"], img_rgb)
        if not st:
            continue
        # 위치: 오른쪽 40% 영역 & 세로 40~75% (얼굴 높이대)
        if st["cx"] < W*0.55 or not (H*0.40 <= st["cy"] <= H*0.75):
            continue
        # 크기: 매우 작음 0.05%~2%
        if not (0.0005*img_area <= st["area"] <= 0.02*img_area):
            continue
        # 어두움: 평균 V 낮음
        if st["mean_v"] > 90:
            continue
        # 원형도 높을수록 좋음(코가 타원형)
        rank = (st["mean_v"], -st["circ"], -st["area"])
        if rank < best_rank:
            best_rank, best = rank, m
    return best

m_chair = pick_chair(masks, img_rgb)
m_nose  = pick_dog_nose(masks, img_rgb)

mask_chair = to_orig(m_chair["segmentation"]) if m_chair else np.zeros((H0,W0), np.uint8)
mask_nose  = to_orig(m_nose["segmentation"])  if m_nose else np.zeros((H0,W0), np.uint8)

# 5) 시각화 + 저장
plt.figure(figsize=(12,6))
plt.subplot(1,2,1); plt.imshow(img_rgb0); plt.imshow(mask_chair, alpha=0.6); plt.axis("off"); plt.title("Chair (Auto SAM)")
plt.subplot(1,2,2); plt.imshow(img_rgb0); plt.imshow(mask_nose , alpha=0.6); plt.axis("off"); plt.title("Dog Nose (Auto SAM)")
plt.tight_layout(); plt.show()

cv2.imwrite("auto_chair_mask.png", mask_chair*255)
cv2.imwrite("auto_dog_nose_mask.png", mask_nose*255)
print("[saved] auto_chair_mask.png, auto_dog_nose_mask.png")
