import astar
import cv2
import numpy as np
from matplotlib import pyplot as plt


def neighbours(node, grid):
    """Return a list of the neighbors of a node in a grid."""
    row, col = node
    height, width = len(grid), len(grid[0])
    candidates = [(row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)]
    return [(r, c) for r, c in candidates
            if 0 <= r < height and 0 <= c < width and not grid[r][c]]


def heuristic(node, goal):
    """Return the Euclidean distance from a node to the goal."""
    x1, y1 = node
    x2, y2 = goal
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_path(start, goal, grid):
    """Returns the path from start to goal in the grid"""
    path = astar.find_path(start, goal, neighbors_fnct=lambda n: neighbours(n, grid), reversePath=False,
                           heuristic_cost_estimate_fnct=heuristic)
    return list(path)

# def main():
#     map_image = cv2.imread("/home/nitesh/workspaces/ros2_mpc_ws/src/ros2_mpc/maps/map_carto.pgm", cv2.IMREAD_GRAYSCALE)
#     map_image[map_image == 0] = 1
#     map_image[map_image > 1] = 0
#     # Dilate the image by 10 pixels
#     kernel = np.ones((10, 10), np.uint8)
#     map_image = cv2.dilate(map_image, kernel, iterations=1)
#     map_image = map_image.astype(np.uint8)
#
#     start = (50, 50)
#     goal = (50, 175)
#     path = list(get_path(start, goal, map_image))
#     print(path)
#
#     plt.imshow(map_image, cmap="gray")
#     plt.plot([x[1] for x in path], [x[0] for x in path])
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()
