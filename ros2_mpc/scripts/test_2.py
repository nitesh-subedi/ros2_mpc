import numpy as np
from rrtplanner import perlin_occupancygrid
from rrtplanner import RRTStar, random_point_og
import yaml
import cv2
from rrtplanner import plot_rrt_lines, plot_path, plot_og, plot_start_goal
import matplotlib.pyplot as plt


def get_points_on_lines(line_segments):
    points = []

    for segment in line_segments:
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        err = dx - dy

        while x1 != x2 or y1 != y2:
            points.append((x1, y1))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        points.append((x2, y2))

    return np.array(points)


def erode_image(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    image = cv2.erode(image, kernel, iterations=1)
    return image.astype(np.uint8)


with open('/home/nitesh/projects/ros2_ws/src/ros2_mpc/maps/map_carto.yaml', 'r') as file:
    params = yaml.safe_load(file)
map_image = cv2.imread('/home/nitesh/projects/ros2_ws/src/ros2_mpc/maps/map_carto.pgm')
# Change image to binary
ret, map_image = cv2.threshold(map_image, params['occupied_thresh'], 255, cv2.THRESH_BINARY)
# Convert it to grayscale
map_image = erode_image(map_image, 5)
map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
map_image[map_image == 0] = 1
map_image[map_image == 255] = 0
og = map_image
# og = perlin_occupancygrid(240, 240, 0.33)
# print(map_image)
n = 800
r_rewire = 100
rrts = RRTStar(og, n, r_rewire)
xstart = np.array([100, 100])
print(xstart)
xgoal = np.array([50, 175])
print(xgoal)
T, gv = rrts.plan(xstart, xgoal)
path = rrts.route2gv(T, gv)
path_pts = rrts.vertices_as_ndarray(T, path)
print(path_pts)
line_points = get_points_on_lines(path_pts)
print(line_points)
# The path is in points connecting straight lines. We need to convert it to a list of points.
# print(path_pts.shape)
# print(type(path_pts))
# print(path_pts)
# plt.imshow(og, cmap='gray')
# plt.show()
# create figure and ax.
fig = plt.figure()
ax = fig.add_subplot()

# these functions alter ax in-place.
plot_og(ax, og)
plot_start_goal(ax, xstart, xgoal)
# plot_rrt_lines(ax, T)
plot_path(ax, path_pts)
# plot_path(ax, np.array(line_points))
plt.scatter(line_points[:, 0], line_points[:, 1])
plt.show()
