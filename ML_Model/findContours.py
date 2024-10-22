import numpy as np
import cv2

# we're going to be inputting a npz matrix of the following dimensions [12 slices, x timepoints, 512, 512]
# we need to output a dictionary (due to potentially differing number of edge points) with the following structure: 
# keys of 12 slices. Each entry is another dictionary with timepoints as the keys and numpy array of edge coords (x, y, z) as the entry

def findContours(predicted_targets):

    edge_coords = {slice: {} for slice in range(predicted_targets.shape[0])}

    theta = np.arange(0, np.pi, np.pi/12) # There are 12 division of the imager holder
             
    for slice_num in edge_coords.keys():
        for time_point in range(predicted_targets.shape[1]):
            
            contours, _ = cv2.findContours(image=predicted_targets[slice_num][time_point],
                                        mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE
                                        )

            # Store the coordinates of all the contours in the coordinates array. 
            slice_coords = np.empty((0, 2), dtype=int)
            for contour in contours:
                slice_coords = np.vstack((slice_coords, contour.reshape(-1, 2)))

            # Apply a rotation, theta, around the x axis
            centroid_x = np.mean(slice_coords[:, 0])

            # Shift the contour to be 'centered' on (0, 0) then perform a rotation around the x axis
            shifted_x = slice_coords[:, 0] - centroid_x
            
            # initialize a z coordinate
            z = np.zeros_like(shifted_x)

            # perform a rotation around the y axis based on the slice
            cos_theta = np.cos(theta[slice_num])
            sin_theta = np.sin(theta[slice_num])
            x_rotated = shifted_x * cos_theta
            
            z_rotated = shifted_x * sin_theta

            # shift the contour back to its origional x centroid after performing the rotation
            reshifted_x = x_rotated + centroid_x
            
            # Reform the slice_coords matrix with the reshifted x, the origional y, and the rotated z
            slice_coords = np.column_stack((reshifted_x, slice_coords[:, 1], z_rotated))

            edge_coords[slice_num][time_point] = slice_coords # add each time point's coords to the correct slice in the dict   
    return edge_coords

if __name__ == '__main__':
    # a brilliant few lines of testing code
    pass