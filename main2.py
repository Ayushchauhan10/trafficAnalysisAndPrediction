import cv2
import numpy as np
import matplotlib.pyplot as plt
from vehicle_counter import VehicleCounter
from traffic_predictor import TrafficPredictor
from config import *
import time
from tensorflow.keras.models import load_model
import joblib


class UnifiedTrafficSystem:
    def __init__(self):
        self.vehicle_counter = VehicleCounter()

        try:
            # Load LSTM model using Keras
            self.traffic_model = load_model(TRAFFIC_MODEL_PATH)
            self.scaler = joblib.load('scaler.save')
            print("Traffic prediction model loaded successfully!")
        except Exception as e:
            print(f"Error loading traffic model: {e}")
            print("Please train the model first by running trainLSTM.py")
            exit(1)

    def predict_traffic(self, vehicle_count):
        """Predict traffic using the LSTM model"""
        # Prepare input features (using dummy temp/rain values)
        features = np.array([[vehicle_count, 20, 0]])  # count, temp=20, rain=0

        # Scale features
        scaled_features = self.scaler.transform(features)

        # Reshape for LSTM (add sequence dimension)
        if len(scaled_features) < SEQ_LENGTH:
            padded = np.zeros((SEQ_LENGTH, scaled_features.shape[1]))
            padded[-len(scaled_features):] = scaled_features
            scaled_features = padded

        scaled_features = scaled_features[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, -1)

        # Make prediction
        prediction = self.traffic_model.predict(scaled_features)[0][0]
        return prediction > THRESHOLD, prediction

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video")
            return

        plt.ion()
        fig, (ax_video, ax_graph) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Traffic Monitoring System', fontsize=16)

        try:
            while cap.isOpened():
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                # Count vehicles
                count, annotated_frame = self.vehicle_counter.process_frame(frame)

                # Predict traffic periodically
                if len(self.vehicle_counter.counts) % (TARGET_FPS * 5) == 0:
                    has_traffic, confidence = self.predict_traffic(count)
                    status = f"Traffic: {'HIGH' if has_traffic else 'LOW'} ({confidence:.2f})"
                    cv2.putText(annotated_frame, status, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display video
                cv2.imshow("Live Traffic", annotated_frame)

                # Update plots
                ax_video.clear()
                ax_video.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                ax_video.set_title('Live Camera Feed')
                ax_video.axis('off')

                ax_graph.clear()
                if len(self.vehicle_counter.counts) > 1:
                    ax_graph.plot(self.vehicle_counter.counts, 'b-')
                    ax_graph.set_title('Vehicle Count Over Time')
                    ax_graph.set_xlabel('Frame')
                    ax_graph.set_ylabel('Count')
                    ax_graph.grid(True)

                plt.pause(0.01)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.close()


if __name__ == "__main__":
    print("Starting Traffic Monitoring System...")
    system = UnifiedTrafficSystem()
    system.run("trafficVideo5Min.mp4")  # or 0 for webcam