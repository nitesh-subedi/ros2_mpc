import numpy as np
import matplotlib.pyplot as plt
import cv2
from ros2_mpc import utils
import yaml


def potential_field_planner(start, goal, obstacles, grid_size, max_iterations=1000, eta=0.09, rho=0.3, alpha=0.5,
                            beta=1.0):
    x = start[0]
    y = start[1]

    path = [start]

    for _ in range(max_iterations):
        dx = goal[0] - x
        dy = goal[1] - y

        if np.sqrt(dx ** 2 + dy ** 2) < grid_size:
            break

        attractive_force = np.array([dx, dy])
        attractive_force = eta * attractive_force / np.linalg.norm(attractive_force)

        repulsive_force = np.zeros(2)

        for obstacle in obstacles:
            ox = obstacle[0]
            oy = obstacle[1]

            distance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)

            if distance < rho:
                repulsive_force[0] += alpha * (1 / distance - 1 / rho) * (x - ox) / distance ** 3
                repulsive_force[1] += alpha * (1 / distance - 1 / rho) * (y - oy) / distance ** 3

        total_force = attractive_force + repulsive_force
        x += beta * total_force[0]
        y += beta * total_force[1]

        path.append([x, y])

    return path


with open('/home/nitesh/projects/ros2_ws/src/ros2_mpc/maps/map_carto.yaml', 'r') as file:
    params = yaml.safe_load(file)
map_image = cv2.imread('/home/nitesh/projects/ros2_ws/src/ros2_mpc/maps/map_carto.pgm')
# Change image to binary
ret, map_image = cv2.threshold(map_image, params['occupied_thresh'], 255, cv2.THRESH_BINARY)
# Convert it to grayscale
map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
print(map_image.shape)
obstacles = np.array(np.where(map_image == 0)).T
print(obstacles)
print(obstacles.shape)
# plt.imshow(map_image)
# plt.show()
map_info = {'resolution': params['resolution'], 'origin': params['origin'][:2]}
start = [0, 0]
goal = [4, 2]
start = utils.world_to_map(start[0], start[1], map_image, map_info)
# Swap x and y
start = [start[1], start[0]]
goal = utils.world_to_map(goal[0], goal[1], map_image, map_info)
goal = [goal[1], goal[0]]
print(start, goal)
# # Example usage
# obstacles = [[5, 5], [7, 8], [3, 6], [2, 3]]
grid_size = 1.0

path = potential_field_planner(start, goal, obstacles, grid_size)
print(path[-1])
#
# Plotting the result
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], '-b')
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'go')
for obstacle in obstacles:
    plt.plot(obstacle[0], obstacle[1], 'ks')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Potential Field Path Planning')
plt.grid(True)
plt.show()
