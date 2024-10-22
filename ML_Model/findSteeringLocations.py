
import pickle
import os
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Enable verbose output for Qhull
os.environ['QHULL_OPTIONS'] = 'Tv'

def findSteeringLocations(shapes_points_list, delta=1):
    # Combine points from all shapes to get the bounding box
    all_points = np.vstack(shapes_points_list)
    min_coords = np.min(all_points, axis=0)  # Minimum x, y, z coordinates
    max_coords = np.max(all_points, axis=0)  # Maximum x, y, z coordinates
    
    # Create a grid within the bounding box
    x_vals = np.arange(min_coords[0], max_coords[0] + delta, delta)
    y_vals = np.arange(min_coords[1], max_coords[1] + delta, delta)
    z_vals = np.arange(min_coords[2], max_coords[2] + delta, delta)
    
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # Create a combined structure of the volume defined by shape edges
    volume_points = np.vstack(shapes_points_list)
    
    # Ensure points are unique
    volume_points = np.unique(volume_points, axis=0)
    
    # Ensure the points are sufficient to form a 3-dimensional hull
    if len(volume_points) < 4:
        raise ValueError('Not enough points to form a tetrahedron (at least 4 points required).')
    
    # Convex hull to test if points are inside the volume
    try:
        hull = Delaunay(volume_points)  # Using Delaunay since it also allows point-in-hull checks
    except RuntimeError as e:
        print("Delaunay triangulation failed. Details:", e)
        raise
    
    # Find points inside the volume defined by shape edges
    inside_points = []
    for point in grid_points:
        if hull.find_simplex(point) >= 0:
            inside_points.append(point)
    
    inside_points = np.array(inside_points)
    return inside_points

def visualize_shapes_and_volume(shapes_points_list, volume_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each shape
    for shape_points in shapes_points_list:
        ax.plot(shape_points[:,0], shape_points[:,1], shape_points[:,2], linestyle='-', linewidth=2, color='blue', alpha=0.2)
    #     if len(shape_points) >= 3:
    #         try:
    #             ax.plot_trisurf(shape_points[:, 0], shape_points[:, 1], shape_points[:, 2], alpha=0.5)
    #         except RuntimeError as e:
    #             print(f"Failed to plot_trisurf for a shape: {e}")

    # Plot the volume grid points
    ax.scatter(volume_points[:, 0], volume_points[:, 1], volume_points[:, 2], color='red', s=5, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == '__main__':

    with open('ideal_edge_coords.pkl', 'rb') as file:
        edge_coords = pickle.load(file)

    shape1_points = edge_coords[0][0]
    shape2_points = edge_coords[1][0]
    shape3_points = edge_coords[2][0]
    shape4_points = edge_coords[3][0]
    shape5_points = edge_coords[4][0]
    shape6_points = edge_coords[5][0]
    shape7_points = edge_coords[6][0]
    shape8_points = edge_coords[7][0]
    shape9_points = edge_coords[8][0]
    shape10_points = edge_coords[9][0]
    shape11_points = edge_coords[10][0]
    shape12_points = edge_coords[11][0]

    shape_points_list = [shape1_points, shape2_points, shape3_points, shape4_points, shape5_points, shape6_points, shape7_points, shape8_points, shape9_points, shape10_points, shape11_points, shape12_points]

    # Generate all points within the volume defined by the two shapes
    delta = 4
    volume_points = findSteeringLocations(shape_points_list, delta)

    # Visualize the shapes and the volume points
    visualize_shapes_and_volume(shape_points_list, volume_points)


