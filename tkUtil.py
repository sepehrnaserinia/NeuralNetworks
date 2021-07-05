import tkinter as tk
from PIL import Image
from PIL import ImageTk

def tk_img(cv_img):
    pil_img = Image.fromarray(cv_img)
    tk_img = ImageTk.PhotoImage(pil_img)
    return tk_img

def panel_img_show(panel, img, place=None):
    if panel is None:
        panel = tk.Label(image=img)
        panel.image = img
        panel.place(x=place[0], y=place[1])
    else:
        panel.configure(image=img)
        panel.image = img
    return panel

def panel_text_show(panel, text, place=None):
    if panel is None:
        panel = tk.Label(text=text, font=("Times New Roman", 11))
        panel.place(x=place[0], y=place[1])
    else:
        panel.configure(text=text)
    return panel

def panel_delete(panel):
    if panel is not None:
        panel.configure(image=None, text="")
        panel.image = None

if __name__ == "__main__":
    print("Header File...")
