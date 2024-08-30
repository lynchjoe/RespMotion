import tkinter as tk


#from User Interface/dataSelection import dataSelection






class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('ASH')
        self.geometry('1600x800+0+0')
        self.resizable(True, True)

        # Below are the variables that need to be used between windows
        

        # Initialize the first frame that allows the user to select the output of the helper script
        #dataSelection(self)

        self.mainloop()

if __name__ == '__main__':
    App()








# dataSelection.py
    # user selects the output of the matlab helper script

# targetSelection.py
    # User is displayed first and last frame of every slice and instructed to draw a box around the target
    # Once the target is selected, the input data is concatenated with the target region to output a preprocessed dataset that can be passed into the ML mode
    # Note here that when the target is selected it may not be on all slices. This could save a good deal of computing power
    
# Model.py
    # identify the target in the selected region and output the masks

# setScale.py
    # temporairly store 1 transformed image and use it to set the scale for the steering locations. IDK if this has to necessairly go here but oh well

# adjustTarget.py
    # Show each slice and let the user confirm the target ID by the model
    # allow the user to scale up or down the treatment on each slice
    # allow the user to redraw the outline of the target if needed. Temporairly retrain the model on the adjusted masks and redisplay the targets
        # iterate here until the user is satisfied
    # THIS MAY BE TOO AMBITIOUS FOR THE NEAR TERM

# steeringLocations.py
    # take the output of the model and calculate the steering locations for the array!
    # output a dictionary of locations for each time point

# treatmentVisualization.py
    # run the liver finder model to get approximate outline of the liver
    # display the liver and the treatment locations


#UI design
# frame 1: select the output of the helper script
# frame 2: draw box around the target
# frame 3: scale treatment region after the model identifies the target
# frame 4: pretty graph of the liver and the steering locations inside it
