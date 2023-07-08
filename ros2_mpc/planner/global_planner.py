import astar
import numpy as np
from rrtplanner import RRTStar
import pyastar2d
from scipy.signal import savgol_filter


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


class AstarGlobalPlanner:
    def __init__(self):
        pass

    @staticmethod
    def neighbours(node, map_image):
        """Return a list of the neighbors of a node in a grid."""
        row, col = node
        height, width = len(map_image), len(map_image[0])
        candidates = [(row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)]
        return [(r, c) for r, c in candidates
                if 0 <= r < height and 0 <= c < width and not map_image[r][c]]

    @staticmethod
    def heuristic(node, goal):
        """Return the Euclidean distance from a node to the goal."""
        x1, y1 = node
        x2, y2 = goal
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_path(self, start, goal, map_image):
        """Returns the path from start to goal in the grid"""
        try:
            path = astar.find_path(start, goal, neighbors_fnct=lambda n: self.neighbours(n, map_image),
                                   reversePath=False,
                                   heuristic_cost_estimate_fnct=self.heuristic)
        except IndexError:
            path = []
        return list(path)


class RRTGlobalPlanner:
    def __init__(self, og):
        self.n = 1200
        self.rewire = 80
        self.og = og
        self.rrts = RRTStar(self.og, self.n, self.rewire)

    def get_path(self, start, goal):
        T, gv = self.rrts.plan(start, goal)
        path = self.rrts.route2gv(T, gv)
        path_pts = self.rrts.vertices_as_ndarray(T, path)
        line_points = get_points_on_lines(path_pts)
        return line_points


class AStarPlanner2:
    def __init__(self):
        self.window_size = 15
        self.poly_degree = 4 
        pass

    def get_path(self, start, goal, map_image):
        map_image[map_image == 1] = 255
        map_image[map_image == 0] = 1
        map_image_astar = map_image.astype(np.float32)
        path = pyastar2d.astar_path(
            map_image_astar, tuple(start), tuple(goal), allow_diagonal=False
        )
        x = np.array(path[:, 1])
        y = np.array(path[:, 0])
        try:
            smoothed_y = savgol_filter(y, self.window_size, self.poly_degree, mode="interp")
        except ValueError:
            smoothed_y = y
        # smooth_x = np.convolve(
        #     x, np.ones(self.window_size) / self.window_size, mode="valid"
        # ).astype(np.int32)
        # smooth_y = np.convolve(
        #     y, np.ones(self.window_size) / self.window_size, mode="valid"
        # ).astype(np.int32)
        return list(zip(smoothed_y, x))

