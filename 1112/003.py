import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Canvas 예제")
root.geometry("400x600")

canvas = tk.Canvas(root, width=400, height=600)
canvas.pack()

# 캔버스 크기 얻기
canvas_width = 400
canvas_height = 600

# 이미지 열고 캔버스 크기에 맞게 리사이즈
pil_img = Image.open("example.jpg")
pil_img = pil_img.resize((canvas_width, canvas_height), Image.LANCZOS)


tk_img = ImageTk.PhotoImage(pil_img)
canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

# 이미지 객체 참조 유지
canvas.image = tk_img

root.mainloop()