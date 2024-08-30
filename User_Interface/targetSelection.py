import os
import tkinter as tk








# initilize a parent application for testing purposes.
class dummyApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Testing ASH')
        self.geometry('1600x800+0+0')
        self.resizable(True, True)

        targetSelection(self)

        self.mainloop()

class targetSelection(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        
