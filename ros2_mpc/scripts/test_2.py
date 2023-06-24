import numpy as np


def navfn(start, goal, costmap):
    """Finds the path with the lowest cost from start to goal in the costmap."""

    # Create a navigation function from the costmap.
    nav_fn = np.zeros_like(costmap)
    for i in range(len(costmap)):
        for j in range(len(costmap[0])):
            nav_fn[i][j] = costmap[i][j]

    # Find the path with the lowest cost from start to goal.
    path = astar(nav_fn, start, goal)

    return path


def astar(nav_fn, start, goal):
    """Finds the shortest path from start to goal in the navigation function."""

    open = []
    closed = []
    open.append(start)

    while len(open) > 0:
        cur = open.pop(0)
        closed.append(cur)

        if cur == goal:
            break

        for neighbor in cur.neighbors:
            if neighbor not in closed:
                open.append(neighbor)

    return closed


if __name__ == "__main__":
    # Create a costmap.
    costmap = np.random.randint(0, 10, (10, 10))

    # Start and goal positions.
    start = (0, 0)
    goal = (9, 9)

    # Find the path with the lowest cost from start to goal.
    path = navfn(start, goal, costmap)

    # Print the path.
    print(path)
