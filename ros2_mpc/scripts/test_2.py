import matplotlib.pyplot as plt
import numpy as np
import pyastar2d
import cv2
import yaml
from ros2_mpc.planner.global_planner import AstarGlobalPlanner

planner = AstarGlobalPlanner()


def erode_image(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    image = cv2.erode(image, kernel, iterations=2)
    return image.astype(np.uint8)


with open('/home/nitesh/projects/ros2_ws/src/ros2_mpc/maps/map_carto.yaml', 'r') as file:
    params = yaml.safe_load(file)
map_image = cv2.imread('/home/nitesh/projects/ros2_ws/src/ros2_mpc/maps/map_carto.pgm')
# Change image to binary
ret, map_image = cv2.threshold(map_image, params['occupied_thresh'], 255, cv2.THRESH_BINARY)
# Convert it to grayscale
map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
map_image = erode_image(map_image, 5)
map_image_ = map_image.copy()
map_image_[map_image_ == 255] = 1
map_image_ = 1 - map_image_
map_image[map_image == 0] = 200
map_image[map_image == 255] = 1
map_image_astar = map_image.astype(np.float32)

# The minimum cost must be 1 for the heuristic to be valid.
# The weights array must have np.float32 dtype to be compatible with the C++ code.
# weights = np.array([[1, 3, 3, 3, 3],
#                     [2, 0, 3, 3, 3],
#                     [2, 2, 0, 3, 3],
#                     [2, 2, 2, 0, 3],
#                     [2, 2, 2, 2, 1]], dtype=np.float32)
# The start and goal coordinates are in matrix coordinates (i, j).
start = (90, 90)
goal = (50, 175)
path = pyastar2d.astar_path(map_image_astar, start, goal, allow_diagonal=True)
x = np.array(path[:, 1])
y = np.array(path[:, 0])
window_size = 10  # Adjust the window size as desired
smooth_x = np.convolve(x, np.ones(window_size) / window_size, mode='valid').astype(np.int32)
smooth_y = np.convolve(y, np.ones(window_size) / window_size, mode='valid').astype(np.int32)
print(list(zip(smooth_x, smooth_y)))
path_new = planner.get_path(start, goal, map_image_)
print(path_new)
plt.imshow(map_image, cmap='gray')
plt.plot(x, y)
plt.plot(smooth_x, smooth_y, label='Smoothed Path')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
