import numpy as np

# create a 2D numpy array
arr = np.array([[50, 60, 70], [80, 90, 55]])

# create a boolean mask for values greater than 65
mask = arr > 65

# use boolean indexing to replace values with 100
arr[mask] = 100

print(arr)
