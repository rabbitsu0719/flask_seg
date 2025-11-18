import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from ultralytics.models.sam import Predictor as SAMPredictor

# 전역 변수 초기화
image = None       # PIL 이미지
photo = None       # tkinter 이미지
masks = []         # 누적 마스크 리스트 (bool numpy 배열)
predictor = None   # SAM Predictor 객체


def init_predictor():
   global predictor
   predictor = SAMPredictor(overrides=dict(model="sam_b.pt"))  # 모델 경로 필요


def load_image():
   global image, photo, masks
   # file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
   file_path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
   if not file_path:
       return
   image = Image.open(file_path).convert("RGBA")
   img_np = np.array(image.convert("RGB"))
   predictor.set_image(img_np)
   masks.clear()
   update_canvas_image()
 


def update_canvas_image():
   global photo
   if image is None:
       return
   display_image = image.copy()
   overlay = Image.new("RGBA", display_image.size, (255, 0, 0, 100))  # 빨간색 반투명
   mask_total = np.zeros((display_image.height, display_image.width), dtype=bool)

   for mask in masks:
       mask_total = np.logical_or(mask_total, mask)
   mask_img = Image.fromarray((mask_total * 255).astype(np.uint8))
   display_image.paste(overlay, mask=mask_img)

   photo = ImageTk.PhotoImage(display_image)
   canvas.delete("all")
   canvas.create_image(0, 0, anchor=tk.NW, image=photo)


def on_click(event):
   global masks
   if image is None:
       return
   x, y = event.x, event.y
   if not (0 <= x < image.width and 0 <= y < image.height):
       return
   points = np.array([[x, y]])
   labels = np.array([1])  # positive point
   results = predictor(points=points, point_labels=labels, multimask_output=False)
   if results and len(results) > 0:
       result = results[0]
       if result.masks is not None:
           mask_np = result.masks.data.cpu().numpy()[0] > 0.5
           masks.append(mask_np)
           update_canvas_image()


# tkinter UI 생성
root = tk.Tk()
root.title("Ultralytics SAM 이미지 마스크 에디터")

load_button = tk.Button(root, text="이미지 로드", command=load_image)
load_button.pack(pady=10)


canvas = tk.Canvas(root, bg="white", width=800, height=600)
canvas.pack()
canvas.bind("<Button-1>", on_click)

init_predictor()
root.mainloop()


