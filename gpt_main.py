import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from vehicle_counter import VehicleCounter
from traffic_predictor import TrafficPredictor
from config import *


class UnifiedTrafficSystem:
    def __init__(self):
        self.vehicle_counter = VehicleCounter()
        try:
            self.predictor = TrafficPredictor().load_models()
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please train models first by running trainLSTM.py")
            exit(1)

        self.forecast_interval = 1  # seconds between predictions
        self.last_forecast_time = 0
        self.all_counts = []  # Store all historical counts
        self.predicted_series = []  # Store continuous predictions (actual + forecast)
        self.prediction_start_idx = -1  # Index where predictions begin
        self.max_display_points = 300  # Limit points for smoother display
        self.fps = 30  # Default fps fallback

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.fps = fps

        plt.ion()
        fig, (ax_video, ax_graph, ax_zoom) = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle('Traffic Monitoring System', fontsize=14)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Count vehicles
                count, annotated_frame = self.vehicle_counter.process_frame(frame)
                self.all_counts.append(count)

                # Update forecast periodically
                current_time = time.time()
                if current_time - self.last_forecast_time > self.forecast_interval:
                    if len(self.all_counts) >= int(SEQ_LENGTH):
                        recent_counts = self.all_counts[-int(SEQ_LENGTH):]
                        new_predictions = self.predictor.predict_future(recent_counts)

                        if self.prediction_start_idx < 0:
                            # First prediction - initialize series
                            self.prediction_start_idx = len(self.all_counts) - 1
                            self.predicted_series = [float('nan')] * self.prediction_start_idx
                            self.predicted_series.extend(new_predictions)
                        else:
                            available_length = len(self.all_counts) - self.prediction_start_idx
                            if available_length < len(self.predicted_series):
                                self.predicted_series[available_length:available_length + FORECAST_STEPS] = new_predictions
                            else:
                                self.predicted_series.extend(new_predictions)

                    self.last_forecast_time = current_time

                # Predict current traffic state
                if len(self.all_counts) > 0:
                    has_traffic, confidence = self.predictor.predict_traffic([self.all_counts[-1]])
                    status = f"Traffic: {'HIGH' if has_traffic else 'LOW'} ({confidence:.2f})"
                    cv2.putText(annotated_frame, status, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display
                cv2.imshow("Live Traffic", annotated_frame)
                self.update_plots(ax_video, ax_graph, ax_zoom, annotated_frame)
                plt.pause(0.01)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.close()

    def update_plots(self, ax_video, ax_graph, ax_zoom, frame):
        ax_video.clear()
        ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_video.set_title('Live Camera Feed')
        ax_video.axis('off')

        ax_graph.clear()
        if len(self.all_counts) > 0:
            # Determine display range
            start_idx = max(0, len(self.all_counts) - self.max_display_points)
            x_values = range(start_idx, len(self.all_counts))
            actual_counts = self.all_counts[start_idx:]

            ax_graph.plot(x_values, actual_counts, 'b-', label='Actual Counts', linewidth=2)

            if self.prediction_start_idx >= 0 and len(self.predicted_series) > 0:
                pred_start = max(start_idx, self.prediction_start_idx)
                pred_x = range(pred_start, pred_start + len(self.predicted_series))
                pred_to_show = self.predicted_series[:len(pred_x)]
                if pred_start < len(self.all_counts) + FORECAST_STEPS:
                    ax_graph.plot(pred_x, pred_to_show, 'r--', label='Predicted', linewidth=2)

            ax_graph.set_title('Traffic Volume Over Time')
            ax_graph.set_xlabel('Frame Number')
            ax_graph.set_ylabel('Vehicle Count')
            ax_graph.legend(loc='upper right')
            ax_graph.grid(True, alpha=0.3)

            y_min = min(actual_counts)
            y_max = max(actual_counts)
            if self.prediction_start_idx >= 0:
                valid_preds = [x for x in self.predicted_series if not np.isnan(x)]
                if valid_preds:
                    y_min = min(y_min, min(valid_preds))
                    y_max = max(y_max, max(valid_preds))

            padding = 0.1 * (y_max - y_min) if y_max != y_min else 1
            ax_graph.set_ylim(max(0, y_min - padding), y_max + padding)

        # === ZOOMED VIEW: Last 10 seconds of data ===
        ax_zoom.clear()
        ax_zoom.set_title('Last 10 Seconds')
        ax_zoom.set_xlabel('Frame')
        ax_zoom.set_ylabel('Vehicle Count')

        frames_to_show = int(self.fps * 10)
        if len(self.all_counts) > 0:
            zoom_start = max(0, len(self.all_counts) - frames_to_show)
            zoom_x = range(zoom_start, len(self.all_counts))
            zoom_y = self.all_counts[zoom_start:]
            ax_zoom.plot(zoom_x, zoom_y, 'g-', label='Recent', linewidth=2)

        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.legend()


if __name__ == "__main__":
    print("Starting Traffic Monitoring System...")
    system = UnifiedTrafficSystem()
    system.run("traffic_prediction1.mp4")  # or use 0 for webcam