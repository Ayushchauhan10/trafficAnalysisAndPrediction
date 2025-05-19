# # Model paths
# VEHICLE_MODEL_PATH = "yolov8n.pt"
# TRAFFIC_MODEL_PATH = "traffic_lstm.h5"
#
# # Video processing
# TARGET_FPS = 15
# GRAPH_HISTORY_SECONDS = 10
#
# # LSTM parameters
# SEQ_LENGTH = 24  # Use 24 historical points (hours)
# THRESHOLD = 0.6  # Decision threshold for traffic/no-traffic
# Model paths
VEHICLE_MODEL_PATH = "yolov8n.pt"
TRAFFIC_MODEL_PATH = "traffic_lstm.h5"
FORECAST_MODEL_PATH = "forecast_lstm.h5"

# Video processing
TARGET_FPS = 15
GRAPH_HISTORY_SECONDS = 10

# LSTM parameters
SEQ_LENGTH = 24  # Must be integer
THRESHOLD = 0.5
FORECAST_STEPS = 6
NORMALIZATION_FACTOR = 325