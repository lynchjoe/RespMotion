import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageSelector:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.rect = None
        self.start_point = None
        self.end_point = None

    def open_image(self):
        # Ask user to select an image file
        file_path = filedialog.askopenfilename(title="Select Image File",
                                            filetypes=[
                                                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.gif"), 
                                                ("All files", "*.*")
                                            ])
        if file_path:
            img = Image.open(file_path).convert("RGB")
            self.image = np.array(img)
        else:
            self.image = None

    def on_mouse_press(self, event):
        # Capture the starting point
        self.start_point = (event.xdata, event.ydata)
        self.rect = plt.Rectangle(self.start_point, 0, 0, fill=False, edgecolor='red')
        self.ax.add_patch(self.rect)

    def on_mouse_release(self, event):
        # Capture the ending point
        self.end_point = (event.xdata, event.ydata)
        print(f"Rectangle selected from {self.start_point} to {self.end_point}")
        
        # Remove the rectangle after selection
        if self.rect:
            self.rect.remove()
        self.rect = None

    def on_mouse_move(self, event):
        if self.start_point and self.rect:
            width = event.xdata - self.start_point[0]
            height = event.ydata - self.start_point[1]
            self.rect.set_width(width)
            self.rect.set_height(height)
            self.rect.set_xy(self.start_point)
            self.fig.canvas.draw()

    def show_image(self):
        self.ax.imshow(self.image)
        self.ax.set_title("Click and drag to select a region")
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        plt.show()

def main():
    # Create a basic Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    selector = ImageSelector()
    selector.open_image()

    if selector.image is not None:
        # Show the image with selection capabilities
        selector.show_image()

# Run the main function
if __name__ == "__main__":
    main()