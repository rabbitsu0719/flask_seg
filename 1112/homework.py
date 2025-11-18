# homework.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from ultralytics.models.sam import Predictor as SAMPredictor


# ----------------------------------------------------------
# SAM 추론기 클래스
# ----------------------------------------------------------
class SamClickPredictor:
    """Ultralytics SAM 기반 클릭 예측 클래스"""

    def __init__(self, model_path="sam_b.pt"):
        self.model_path = model_path
        self.model = SAMPredictor(overrides=dict(model=model_path))
        self.image_np = None
        self.points = []
        self.labels = []

    def load_image(self, img_np):
        """이미지 로드 및 모델 세팅"""
        self.image_np = img_np
        self.model.set_image(img_np)
        self.points.clear()
        self.labels.clear()

    def add_click(self, x, y, positive=True):
        """클릭 포인트 추가 (양수/음수) -> 원본 좌표 기준"""
        self.points.append([int(x), int(y)])
        self.labels.append(1 if positive else 0)

    def predict_mask(self):
        """SAM 마스크 예측 (최대 면적 마스크 선택 + 빈 결과 방어)"""
        if not self.points:
            raise ValueError("최소 1개의 클릭 포인트가 필요합니다.")

        results = self.model(
            points=np.array(self.points, dtype=np.int32),
            point_labels=np.array(self.labels, dtype=np.int32),
            multimask_output=True,       # 여러 후보 생성
            retina_masks=True
        )

        if not results or results[0].masks is None:
            raise RuntimeError("SAM이 마스크를 생성하지 못했습니다. 포인트를 조정해보세요.")

        masks_t = results[0].masks.data  # (N, H, W) or (0, H, W)
        if masks_t.numel() == 0:
            raise RuntimeError("빈 마스크가 반환되었습니다. 다른 위치를 클릭해보세요.")

        masks = masks_t.cpu().numpy() > 0.5
        # 가장 큰 마스크 1개 선택
        best = None
        best_area = -1
        for m in masks:
            area = m.sum()
            if area > best_area:
                best = m
                best_area = area

        return best.astype(np.uint8)  # (H,W) 0/1


# ----------------------------------------------------------
# Tkinter UI
# ----------------------------------------------------------
class SamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📌 SAM Interactive Masking (homework.py)")
        self.predictor = SamClickPredictor("sam_b.pt")
        self.image = None          # PIL RGB 원본
        self.image_np = None       # numpy 원본 (H,W,3)
        self.photo = None
        self.mask_total = None     # 누적 마스크 (H,W) uint8
        self.scale = 1.0           # 미리보기 스케일

        # 버튼 영역
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="이미지 열기", command=self.load_image).pack(side="left", padx=5)
        tk.Button(btn_frame, text="리셋", command=self.reset).pack(side="left", padx=5)
        tk.Button(btn_frame, text="결과 저장", command=self.save_result).pack(side="left", padx=5)

        # 캔버스(고정 크기 미리보기)
        self.canvas_w, self.canvas_h = 800, 600
        self.canvas = tk.Canvas(root, bg="white", width=self.canvas_w, height=self.canvas_h)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_left_click)   # 양수 클릭
        self.canvas.bind("<Button-3>", self.on_right_click)  # 음수 클릭

        self.status = tk.Label(root, text="이미지를 열고 클릭하세요", anchor="w")
        self.status.pack(fill="x")

    # -----------------------------------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        self.image = Image.open(path).convert("RGB")
        self.image_np = np.array(self.image)

        # SAM에 원본 세팅
        self.predictor.load_image(self.image_np)

        # 누적 마스크 초기화
        H, W = self.image_np.shape[:2]
        self.mask_total = np.zeros((H, W), dtype=np.uint8)

        # 미리보기 스케일 계산 (캔버스 안에 맞게)
        self.scale = min(self.canvas_w / W, self.canvas_h / H, 1.0)

        # 미리보기 업데이트
        self.update_canvas(self.image_np)
        self.status.config(text=f"로드 완료: {os.path.basename(path)} (좌클릭=전경, 우클릭=배경)")

    # -----------------------------------
    def to_display(self, img_np):
        """원본 -> 미리보기 크기로 리사이즈"""
        if img_np is None:
            return None
        H, W = img_np.shape[:2]
        if self.scale != 1.0:
            dW, dH = int(W * self.scale), int(H * self.scale)
            img_np = cv2.resize(img_np, (dW, dH), interpolation=cv2.INTER_LINEAR)
        return img_np

    def to_image_coords(self, x_disp, y_disp):
        """미리보기 좌표 -> 원본 좌표로 역변환 (오프셋 없이 좌상단 기준)"""
        ix = int(round(x_disp / self.scale))
        iy = int(round(y_disp / self.scale))
        return ix, iy

    def update_canvas(self, img_np):
        disp = self.to_display(img_np)
        img = Image.fromarray(disp)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        # 좌상단(0,0)에 배치
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    # -----------------------------------
    def on_left_click(self, event):
        self.handle_click(event.x, event.y, positive=True)

    def on_right_click(self, event):
        self.handle_click(event.x, event.y, positive=False)

    def handle_click(self, x_disp, y_disp, positive=True):
        if self.image_np is None:
            return

        # 미리보기 영역 밖 클릭 방지
        dH, dW = int(self.image_np.shape[0] * self.scale), int(self.image_np.shape[1] * self.scale)
        if not (0 <= x_disp < dW and 0 <= y_disp < dH):
            self.status.config(text="이미지 영역 밖을 클릭했습니다.")
            return

        # 원본 좌표로 변환
        ix, iy = self.to_image_coords(x_disp, y_disp)

        # 범위 체크
        H, W = self.image_np.shape[:2]
        if not (0 <= ix < W and 0 <= iy < H):
            self.status.config(text="원본 좌표 범위를 벗어났습니다.")
            return

        # SAM 클릭 추가 & 예측
        self.predictor.add_click(ix, iy, positive=positive)
        try:
            mask = self.predictor.predict_mask()  # (H,W) 0/1
        except Exception as e:
            message = f"예측 실패: {e}"
            self.status.config(text=message)
            return

        # 누적
        self.mask_total = np.logical_or(self.mask_total, mask).astype(np.uint8)

        # 오버레이 시각화 (원본 기준)
        vis = self.overlay_mask(self.image_np, mask, color=(255, 0, 0))
        self.update_canvas(vis)
        self.status.config(text=f"클릭 ({ix},{iy}) → {'전경' if positive else '배경'} 지정")

    # -----------------------------------
    @staticmethod
    def overlay_mask(image_np, mask01, color=(255, 0, 0), alpha=0.55):
        """원본에 마스크 색상 오버레이 후 반환"""
        vis = image_np.copy()
        m = mask01.astype(bool)
        # 단순 치환(선명) + 알파블렌딩 혼합
        overlay = vis.copy()
        overlay[m] = color
        vis = (alpha * overlay + (1 - alpha) * vis).astype(np.uint8)
        return vis

    def reset(self):
        if self.image_np is None:
            return
        self.predictor.points.clear()
        self.predictor.labels.clear()
        self.mask_total.fill(0)
        self.update_canvas(self.image_np)
        self.status.config(text="포인트/마스크 초기화 완료")

    # -----------------------------------
    def save_result(self):
        if self.image_np is None:
            messagebox.showwarning("경고", "저장할 이미지가 없습니다.")
            return
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)

        # 마스크 (0/255)
        mask_img = (self.mask_total * 255).astype(np.uint8)
        mask_path = os.path.join(out_dir, "mask.png")
        cv2.imwrite(mask_path, mask_img)

        # 오버레이 시각화 저장
        vis = self.overlay_mask(self.image_np, self.mask_total, color=(255, 0, 0))
        vis_path = os.path.join(out_dir, "vis.png")
        cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        self.status.config(text=f"저장 완료 → {mask_path}, {vis_path}")


# ----------------------------------------------------------
# 실행
# ----------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SamApp(root)
    root.mainloop()
