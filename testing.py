import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import sys

# use this script for showing 3d plots

with open('ideal_edge_coords.pkl', 'rb') as file:
    edge_coords = pickle.load(file)

with open('steering_coords.pkl', 'rb') as file:
    steering_coords = pickle.load(file)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# for position in edge_coords.keys():
#     ax.scatter(edge_coords[position][28][:, 0], edge_coords[position][28][:, 1], edge_coords[position][28][:, 2], s=1, c='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.legend(['Slice 1', 'Slice 2', 'Slice 3', 'Slice 4', 'Slice 5', 'Slice 6', 'Slice 7', 'Slice 8', 'Slice 9', 'Slice 10', 'Slice 11', 'Slice 12'])
ax.set_xlim(-20, 20)
ax.set_ylim(235, 280)
ax.set_zlim(200, 265)

i = 0
for position in edge_coords.keys():

    if i % 2 == 0:            
        ax.plot(edge_coords[position][0][:, 2], edge_coords[position][0][:, 0], edge_coords[position][0][:, 1], linestyle='-', linewidth=2, color='blue', alpha=0.2, label='Target')
        ax.plot(edge_coords[position][15][:, 2], edge_coords[position][15][:, 0], edge_coords[position][15][:, 1], linestyle='-', linewidth=2, color='red', alpha=0.2)
    i += 1

# ax.scatter(steering_coords[0][:, 2], steering_coords[0][:, 0], steering_coords[0][:, 1], color='blue', s=5, alpha=0.5, label='Exhale')
# ax.scatter(steering_coords[15][:, 2], steering_coords[15][:, 0], steering_coords[15][:, 1], color='red', s=5, alpha=0.5, label='Steering Locations')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()


sys.exit()

def plot_slices(time_point):

    ax.clear()
    
    for position in edge_coords.keys():

        ax.plot(edge_coords[position][time_point][:, 0], edge_coords[position][time_point][:, 1], edge_coords[position][time_point][:, 2], linestyle='-', linewidth=2, color='blue', alpha=0.2)

    ax.scatter(steering_coords[time_point][:, 0], steering_coords[time_point][:, 1], steering_coords[time_point][:, 2], color='red', s=5, alpha=0.5)



    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(200, 300)
    ax.set_ylim(200, 300)
    ax.set_zlim(-50, 50)

def update(frame): # Note here that "frame" is matplotlib's animation variable name. In the animation function below, the number of frames is defined which can be thought of as the number of time_points. The update function is effectively iterating thru the time points
    plot_slices(frame)

num_frames = 30 # (slices, time points, im_height, im_width)
ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=True, interval=2000) # as mentioned earlier, "frames" specifies how many time points there are.
ani.save('Demo.gif', writer=animation.PillowWriter(fps=15))
# plt.show()