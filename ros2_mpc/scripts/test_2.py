import matplotlib.pyplot as plt
import numpy as np
import heapq

# Define the map size and resolution
MAP_WIDTH = 10
MAP_HEIGHT = 10
MAP_RESOLUTION = 1.0

# Define the start and goal positions (grid coordinates)
start = (0, 0)
goal = (9, 9)

# Define the grid map with obstacle information
obstacle_map = np.zeros((MAP_HEIGHT, MAP_WIDTH))
obstacle_map[3:7, 4] = 1
obstacle_map[4, 1:9] = 1

# Define the costmap based on the obstacle information
costmap = np.where(obstacle_map == 1, np.inf, 1)


# Define the heuristic function (Euclidean distance)
def heuristic(node):
    dx = node[0] - goal[0]
    dy = node[1] - goal[1]
    return np.sqrt(dx ** 2 + dy ** 2)


# Perform the A* search
open_list = []
heapq.heappush(open_list, (0, start))
parent = {start: None}
cost_so_far = {start: 0}

while open_list:
    current_cost, current_node = heapq.heappop(open_list)

    if current_node == goal:
        break

    neighbors = [
        (current_node[0] - 1, current_node[1]),  # left
        (current_node[0] + 1, current_node[1]),  # right
        (current_node[0], current_node[1] - 1),  # down
        (current_node[0], current_node[1] + 1)  # up
    ]

    for neighbor in neighbors:
        if neighbor[0] < 0 or neighbor[0] >= MAP_WIDTH or \
                neighbor[1] < 0 or neighbor[1] >= MAP_HEIGHT:
            continue

        new_cost = cost_so_far[current_node] + costmap[neighbor[1], neighbor[0]]
        if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
            cost_so_far[neighbor] = new_cost
            priority = new_cost + heuristic(neighbor)
            heapq.heappush(open_list, (priority, neighbor))
            parent[neighbor] = current_node

# Reconstruct the path
path = []
current_node = goal
while current_node:
    path.append(current_node)
    current_node = parent[current_node]

path.reverse()

# Print the generated path
print("Generated Path:")
# for node in path:
#     print(node)

plt.plot([node[0] for node in path], [node[1] for node in path], 'b-')
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'ro')
plt.plot(obstacle_map, 'k-')
plt.show()
