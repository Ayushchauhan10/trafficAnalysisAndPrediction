import cv2
import time
from collections import deque
from ultralytics import YOLO
from config import VEHICLE_MODEL_PATH, TARGET_FPS, GRAPH_HISTORY_SECONDS


class VehicleCounter:
    def __init__(self):
        self.model = YOLO(VEHICLE_MODEL_PATH)
        self.model.fuse()
        self.counts = deque(maxlen=int(GRAPH_HISTORY_SECONDS * TARGET_FPS))  # Ensure integer
        self.timestamps = deque(maxlen=int(GRAPH_HISTORY_SECONDS * TARGET_FPS))

    def process_frame(self, frame):
        results = self.model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
        count = len(results[0].boxes) if results[0].boxes is not None else 0
        self.counts.append(count)
        self.timestamps.append(time.time())
        return count, results[0].plot()



# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import deque
# import time
# from config import VEHICLE_MODEL_PATH, TARGET_FPS, GRAPH_HISTORY_SECONDS
#
#
# class VehicleCounter:
#     def __init__(self):
#         self.model = YOLO(VEHICLE_MODEL_PATH)
#         self.model.fuse()
#         self.counts = deque(maxlen=GRAPH_HISTORY_SECONDS * TARGET_FPS)
#         self.timestamps = deque(maxlen=GRAPH_HISTORY_SECONDS * TARGET_FPS)
#
#     def process_frame(self, frame):
#         results = self.model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
#         count = len(results[0].boxes) if results[0].boxes is not None else 0
#         self.counts.append(count)
#         self.timestamps.append(time.time())
#         return count, results[0].plot()