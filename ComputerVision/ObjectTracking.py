from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from typing import List, Tuple


class CentroidTracker:
    """
    This class implements a centroid based object tracking approach.
    Taken from https://github.com/lev1khachatryan/Centroid-Based_Object_Tracking
    """

    def __init__(self, max_disappeared: int = 15):
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid: List[int]):
        # Use the next available object ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, object_id: int):
        # Deregister an object by deleting both entries in respective dictionaries
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> OrderedDict[int, List[int]]:
        if len(rects) == 0:
            # Loop over any existing tracked objects and mark them as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # Deregister objects if they reached the number of maximum disappeared frames
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Initialize an array of input centroids for the current frame and loop over the bounding box rectangles
        input_centroids = np.zeros((len(rects), 2), dtype=np.int)
        for (i, (startX, startY, width, height)) in enumerate(rects):
            # Use the bounding box coordinates to derive the centroid
            c_x = startX + int(width / 2)
            c_y = startY + int(height / 2)
            input_centroids[i] = (c_x, c_y)

        # Register all objects in case there are no tracked objects yet
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])

        # Try to match input centroid to tracked objects
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute the distance between each pair of object centroids and input centroids
            d = dist.cdist(np.array(object_centroids), input_centroids)

            # Find the smallest value in each row and sort the row indexes based on their minimum values
            rows = d.min(axis=1).argsort()

            # Finding the smallest value in each column and then sort using the previously computed row index list
            cols = d.argmin(axis=1)[rows]

            # Create sets to keep track of which of the rows and column indexes already examined
            used_rows = set()
            used_cols = set()

            # Loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # Ignore combination if row or col were already used
                if row in used_rows or col in used_cols:
                    continue

                # Grab the object ID for the current row, set its new centroid, and reset the disappeared counter
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # Mark row and column as used
                used_rows.add(row)
                used_cols.add(col)

            # Compute both the row and column index not examined yet
            unused_rows = set(range(0, d.shape[0])).difference(used_rows)
            unused_cols = set(range(0, d.shape[1])).difference(used_cols)

            # Check if objects have disappeared in case of more tracked objects than input objects
            if d.shape[0] >= d.shape[1]:
                # Loop over the unused row indexes
                for row in unused_rows:
                    # Grab the object ID for the corresponding row index and increment the disappeared counter
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    # Deregister objects that have reached the maximum number of disappeared frames
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # Register each input centroid as new trackable object in case of more input centroids than tracked objects
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        # Return the set of trackable objects
        return self.objects
