
from pathlib import Path


from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/Users/mostafaamgad/Desktop/build/assets/frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1280x720")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 720,
    width = 1280,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    640.0,
    317.0,
    image=image_image_1
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=457.0,
    y=337.0,
    width=121.66667175292969,
    height=46.74561309814453
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=578.6666259765625,
    y=337.0,
    width=121.66667175292969,
    height=46.74561309814453
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_3 clicked"),
    relief="flat"
)
button_3.place(
    x=700.3333740234375,
    y=337.0,
    width=121.66667175292969,
    height=46.74561309814453
)

canvas.create_text(
    499.9035339355469,
    351.0877685546875,
    anchor="nw",
    text="SVM",
    fill="#000000",
    font=("Inter SemiBold", 15 * -1)
)

canvas.create_text(
    629.8947143554688,
    351.0877685546875,
    anchor="nw",
    text="LR",
    fill="#000000",
    font=("Inter SemiBold", 16 * -1)
)

canvas.create_text(
    750.2806396484375,
    351.0877685546875,
    anchor="nw",
    text="NB",
    fill="#000000",
    font=("Inter SemiBold", 16 * -1)
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    640.0,
    645.5,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=290.0,
    y=592.0,
    width=700.0,
    height=105.0
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_4 clicked"),
    relief="flat"
)
button_4.place(
    x=161.0,
    y=604.0,
    width=84.0,
    height=84.0
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    202.33331298828125,
    646.333251953125,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    202.33331298828125,
    646.333251953125,
    image=image_image_3
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_5 clicked"),
    relief="flat"
)
button_5.place(
    x=184.0,
    y=628.0,
    width=37.0,
    height=37.0
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    356.0,
    618.5,
    image=entry_image_2
)
entry_2 = Text(
    bd=0,
    bg="#000000",
    fg="#000716",
    highlightthickness=0
)
entry_2.place(
    x=290.0,
    y=604.0,
    width=132.0,
    height=27.0
)
window.resizable(False, False)
window.mainloop()
