import astar
import numpy as np


class GlobalPlanner:
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
        path = astar.find_path(start, goal, neighbors_fnct=lambda n: self.neighbours(n, map_image), reversePath=False,
                               heuristic_cost_estimate_fnct=self.heuristic)
        return list(path)
