import heapq
import cv2


class DStar:
    def __init__(self, start, goal, grid):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.g = {}
        self.rhs = {}
        self.U = []
        self.km = 0
        self.g[start] = 0
        self.rhs[start] = self.heuristic(start, goal)
        heapq.heappush(self.U, (self.calculate_key(start), start))

    def calculate_key(self, node):
        return (min(self.g.get(node, float('inf')), self.rhs.get(node, float('inf'))) + self.heuristic(node,
                                                                                                       self.goal) + self.km,
                min(self.g.get(node, float('inf')), self.rhs.get(node, float('inf'))))

    def update_vertex(self, node):
        if node != self.start:
            self.rhs[node] = min(self.heuristic(node, n) + self.g[n] for n in self.neighbours(node))
        if node in self.U:
            self.U.remove(node)
        if self.g.get(node, float('inf')) != self.rhs.get(node, float('inf')):
            heapq.heappush(self.U, (self.calculate_key(node), node))

    def compute_shortest_path(self):
        while self.U and (self.U[0][0] < self.calculate_key(self.goal) or self.rhs[self.goal] != self.g[self.goal]):
            k_old, node = heapq.heappop(self.U)
            k_new = self.calculate_key(node)
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, node))
            elif self.g.get(node, float('inf')) > self.rhs.get(node, float('inf')):
                self.g[node] = self.rhs[node]
                for n in self.neighbours(node):
                    self.update_vertex(n)
            else:
                self.g[node] = float('inf')
                for n in self.neighbours(node):
                    self.update_vertex(n)
                self.update_vertex(node)

    def get_shortest_path(self):
        path = [self.start]
        while path[-1] != self.goal:
            node = min(self.neighbours(path[-1]),
                       key=lambda n: self.heuristic(n, self.goal) + self.g.get(n, float('inf')))
            path.append(node)
        return path

    @staticmethod
    def heuristic(node, goal):
        # Euclidean distance heuristic
        return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2) ** 0.5

    def neighbours(self, node):
        """Return a list of the neighbours of a node in a grid."""
        grid = self.grid
        row, col = node
        height, width = len(grid), len(grid[0])
        candidates = [(row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)]
        return [(r, c) for r, c in candidates
                if 0 <= r < height and 0 <= c < width and not grid[r][c]]


start = (0, 0)
goal = (5, 5)
map_image = cv2.imread("/home/nitesh/workspaces/ros2_mpc_ws/src/ros2_mpc/maps/map_carto.pgm", cv2.IMREAD_GRAYSCALE)
map_image[map_image == 0] = 1
map_image[map_image > 1] = 0

dstar = DStar(start, goal, map_image)
dstar.compute_shortest_path()
