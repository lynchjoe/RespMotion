import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from targetSelection import targetSelection


# initilize a parent application for testing purposes.
class dummyApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Testing ASH')
        self.geometry('1600x800+0+0')
        self.resizable(True, True)

        dataSelection(self)

        self.mainloop()


class dataSelection(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.data_path = None

        self.pickData = tk.Button(master=self, text='Select Ultrasound Data', command=self.loadData)
        
        self.outputLabel = tk.Label(text='test')
        
        self.outputLabel.pack(side='top', pady=20, fill='both')
        self.pickData.pack(side='top', pady=20)

        

        self.pack(fill='both')
        
    def loadData(self):
        initial_dir = '/Users/joel/Files/VS Code/Final Gating App/'
        self.data_path = filedialog.askdirectory(initialdir=initial_dir)

        # if a file is chosen, run some checks to make sure it is compatable with the model
        if self.data_path:
            positions = os.listdir(self.data_path)
            positions.remove('.DS_Store')
            passed_tests = True

            # Check that there are the correct number of positions
            if len(positions) != 12:
                self.outputLabel.configure(text=f'The data contain {len(positions)} positions. The model expects 12.')
                passed_tests = False

            # Check that there are the correct file types
            exit_loops = False
            for position in positions:
                if exit_loops:
                    break

                current_position = os.path.join(self.data_path, position)
                for file in os.listdir(current_position):
                    file_path = os.path.join(current_position, file)
                    

                    if os.path.isfile(file_path):
                        _, file_extension = os.path.splitext(file)
                        if file_extension != '.png':
                            self.outputLabel.configure(text=f'Unexpected file type: {file_extension}. Expected .png')
                            break
                            exit_loops = True
                    else:
                        self.outputLabel.configure(text=f'Unexpected item in {position}: {file}')
                        exit_loops = True
                        break

        if passed_tests:
            self.outputLabel.configure(text='Selected Data is in expected format! Select the target region.')
            for widget in self.winfor_children():
                widget.destroy()
            self.destroy()
            targetSelection(self.parent)



if __name__ == '__main__':
    dummyApp()