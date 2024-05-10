import tkinter as tk
import random

def change_label_text():
    # Generate a random color
    hex_color = generate_random_color()
    # Update the label text to the new color code
    label.config(text=f"Background color: {hex_color}")
    # Change the background to the new color
    change_background(hex_color)

def generate_random_color():
    # Generate a random hex color code
    return "#" + "".join([random.choice('0123456789ABCDEF') for _ in range(6)])

def change_background(color):
    # Change the background color of the root window
    root.config(bg=color)

# Create the main window
root = tk.Tk()
root.title("Colorful Tkinter Example")

# Create a label widget
label = tk.Label(root, text="Press the button to change color")
label.pack()

# Create a button widget
button = tk.Button(root, text="Click me", command=change_label_text)
button.pack()

# Run the application
root.mainloop()