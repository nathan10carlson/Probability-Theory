import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define cube vertices
vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                     [0,0,1],[1,0,1],[1,1,1],[0,1,1]])

# Define cube faces
faces = [[vertices[j] for j in [0,1,2,3]],  # bottom
         [vertices[j] for j in [4,5,6,7]],  # top
         [vertices[j] for j in [0,1,5,4]],  # front
         [vertices[j] for j in [2,3,7,6]],  # back
         [vertices[j] for j in [0,3,7,4]],  # left
         [vertices[j] for j in [1,2,6,5]]]  # right

# Plot cube
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='k', alpha=0.2))

# Example geodesic: start at bottom front left, go to top back right
# For simplicity, we draw straight segments across faces manually
geodesic_points = np.array([
    [0,0,0],   # bottom front left
    [0.5,0.5,0], # across bottom face
    [1,1,0.5], # cross to side face
    [1,1,1]   # top back right
])

ax.plot(geodesic_points[:,0], geodesic_points[:,1], geodesic_points[:,2], color='r', linewidth=3, label='Geodesic (approx)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.legend()
plt.show()