import tkinter as tk

root = tk.Tk()

root.geometry("500x400")         # 창 크기 설정
root.minsize(300, 200)           # 최소 크기 제한
root.maxsize(700, 600)           # 최대 크기 제한
root.resizable(True, False)      # 가로 크기 조절 가능, 세로는 불가능

root.title("Label 예제")

def button_clicked():
   label.config(text="버튼이 클릭되었습니다!")

def print_input():
   user_input = entry.get()
   print("입력된 텍스트:", user_input)

# 레이블 생성
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()

# 버튼 생성
button = tk.Button(root, text="클릭하세요", command=button_clicked)
button.pack()

# 한 줄 텍스트 입력
entry = tk.Entry(root)
entry.pack()
button2 = tk.Button(root, text="출력", command=print_input)
button2.pack()

root.mainloop()