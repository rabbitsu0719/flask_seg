import tkinter as tk

root = tk.Tk()
root.title("Label 예제")

label = tk.Label(root, text="Hello, Tkinter!")
label.pack()

root.mainloop()