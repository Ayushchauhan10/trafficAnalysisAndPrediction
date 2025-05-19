# main.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from vehicle_counter import VehicleCounter
from traffic_predictor import TrafficPredictor
from metrics import TrafficMetrics
from config import *


class UnifiedTrafficSystem:
    def __init__(self):
        self.vehicle_counter = VehicleCounter()
        self.metrics = TrafficMetrics()  # Initialize metrics tracker
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
        self.timestamps = []  # Store timestamps for each count
        self.predicted_series = []  # Store continuous predictions
        self.prediction_start_idx = -1  # Index where predictions begin
        self.max_display_points = 300  # Limit points for full timeline
        self.short_window_sec = 10  # Last 10 seconds for the second graph

        # For metrics visualization
        self.show_metrics = False
        self.metrics_fig = None
        self.metrics_ax = None

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video")
            return

        plt.ion()
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2)
        ax_video = fig.add_subplot(gs[:, 0])
        ax_full = fig.add_subplot(gs[0, 1])
        ax_short = fig.add_subplot(gs[1, 1])
        fig.suptitle('Traffic Monitoring System', fontsize=14)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()
                # Count vehicles
                count, annotated_frame = self.vehicle_counter.process_frame(frame)
                self.all_counts.append(count)
                self.timestamps.append(current_time)

                # Update forecast periodically
                if current_time - self.last_forecast_time > self.forecast_interval:
                    if len(self.all_counts) >= int(SEQ_LENGTH):
                        recent_counts = self.all_counts[-int(SEQ_LENGTH):]
                        new_predictions = self.predictor.predict_future(recent_counts)

                        if self.prediction_start_idx < 0:
                            # First prediction
                            self.prediction_start_idx = len(self.all_counts) - 1
                            self.predicted_series = [float('nan')] * self.prediction_start_idx
                            self.predicted_series.extend(new_predictions)
                        else:
                            # Update existing predictions
                            available_length = len(self.all_counts) - self.prediction_start_idx
                            if available_length < len(self.predicted_series):
                                self.predicted_series[
                                available_length:available_length + FORECAST_STEPS] = new_predictions
                            else:
                                self.predicted_series.extend(new_predictions)

                        # Record metrics for regression
                        if len(new_predictions) > 0:
                            # For simplicity, compare first prediction with actual count when available
                            if len(self.all_counts) > self.prediction_start_idx + 1:
                                actual = self.all_counts[self.prediction_start_idx + 1]
                                self.metrics.add_regression_result(actual, new_predictions[0])

                    self.last_forecast_time = current_time

                # Predict current traffic state
                if len(self.all_counts) > 0:
                    has_traffic, confidence = self.predictor.predict_traffic([self.all_counts[-1]])
                    status = f"Traffic: {'HIGH' if has_traffic else 'LOW'} ({confidence:.2f})"
                    cv2.putText(annotated_frame, status, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Record classification metrics (assuming ground truth is not available in real-time)
                    # In a real system, you might get this from another source
                    # For demo purposes, we'll skip adding to metrics here

                # Display
                cv2.imshow("Live Traffic", annotated_frame)
                self.update_plots(ax_video, ax_full, ax_short, annotated_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):  # Toggle metrics display
                    self.toggle_metrics_display()

                plt.pause(0.01)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.close()
            if self.metrics_fig is not None:
                plt.close(self.metrics_fig)

            # Print final metrics
            print("\nFinal Performance Metrics:")
            self.metrics.print_all_metrics()

    def toggle_metrics_display(self):
        """Toggle the metrics visualization window"""
        self.show_metrics = not self.show_metrics

        if self.show_metrics:
            self.display_metrics()
        elif self.metrics_fig is not None:
            plt.close(self.metrics_fig)
            self.metrics_fig = None

    def display_metrics(self):
        """Show a separate window with performance metrics visualization"""
        if self.metrics_fig is None:
            self.metrics_fig, self.metrics_ax = plt.subplots(2, 2, figsize=(14, 10))
            self.metrics_fig.suptitle('Performance Metrics', fontsize=14)
            plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # Get current metrics
        cls_metrics = self.metrics.get_classification_metrics()
        reg_metrics = self.metrics.get_regression_metrics()

        # Clear all axes
        for ax in self.metrics_ax.flat:
            ax.clear()

        # Plot classification metrics if available
        if cls_metrics:
            # Confusion matrix
            cm = cls_metrics['confusion_matrix']
            self.metrics_ax[0, 0].matshow(cm, cmap='Blues', alpha=0.7)
            for (i, j), val in np.ndenumerate(cm):
                self.metrics_ax[0, 0].text(j, i, str(val), ha='center', va='center')
            self.metrics_ax[0, 0].set_title('Confusion Matrix')
            self.metrics_ax[0, 0].set_xticks([0, 1])
            self.metrics_ax[0, 0].set_yticks([0, 1])
            self.metrics_ax[0, 0].set_xticklabels(['Low', 'High'])
            self.metrics_ax[0, 0].set_yticklabels(['Low', 'High'])
            self.metrics_ax[0, 0].set_xlabel('Predicted')
            self.metrics_ax[0, 0].set_ylabel('Actual')

            # Metrics bar chart
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
            metrics_values = [
                cls_metrics['accuracy'],
                cls_metrics['precision'],
                cls_metrics['recall'],
                cls_metrics['f1']
            ]
            self.metrics_ax[0, 1].bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
            self.metrics_ax[0, 1].set_title('Classification Metrics')
            self.metrics_ax[0, 1].set_ylim(0, 1)
            for i, v in enumerate(metrics_values):
                self.metrics_ax[0, 1].text(i, v + 0.02, f"{v:.2f}", ha='center')

        # Plot regression metrics if available
        if reg_metrics:
            # Error distribution
            errors = reg_metrics['error_distribution']
            if errors:
                self.metrics_ax[1, 0].hist(errors, bins=20, color='purple', alpha=0.7)
                self.metrics_ax[1, 0].axvline(0, color='black', linestyle='--')
                self.metrics_ax[1, 0].set_title('Prediction Error Distribution')
                self.metrics_ax[1, 0].set_xlabel('Prediction Error (Predicted - Actual)')
                self.metrics_ax[1, 0].set_ylabel('Frequency')

            # Metrics comparison
            metrics_names = ['MAE', 'RMSE', 'R²']
            metrics_values = [
                reg_metrics['mae'],
                reg_metrics['rmse'],
                max(0, reg_metrics['r2'])  # R² can be negative, but we'll show as 0 for display
            ]
            colors = ['blue' if val != metrics_values[-1] else 'green' for val in metrics_values]
            self.metrics_ax[1, 1].bar(metrics_names, metrics_values, color=colors)
            self.metrics_ax[1, 1].set_title('Regression Metrics')
            for i, v in enumerate(metrics_values):
                self.metrics_ax[1, 1].text(i, v + 0.02, f"{v:.2f}", ha='center')

        # If no metrics available yet
        if not cls_metrics and not reg_metrics:
            self.metrics_ax[0, 0].text(0.5, 0.5, 'No metrics data available yet',
                                       ha='center', va='center')
            for ax in self.metrics_ax.flat[1:]:
                ax.axis('off')

        plt.pause(0.01)

    def update_plots(self, ax_video, ax_full, ax_short, frame):
        ax_video.clear()
        ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_video.set_title('Live Camera Feed')
        ax_video.axis('off')

        # Full timeline plot
        ax_full.clear()
        if len(self.all_counts) > 0:
            # Determine display range
            start_idx = max(0, len(self.all_counts) - self.max_display_points)
            x_values = range(start_idx, len(self.all_counts))
            actual_counts = self.all_counts[start_idx:]

            # Plot actual counts
            ax_full.plot(x_values, actual_counts,
                         'b-', label='Actual Counts', linewidth=2)

            # Plot predictions if available
            if self.prediction_start_idx >= 0 and len(self.predicted_series) > 0:
                pred_start = max(start_idx, self.prediction_start_idx)
                pred_x = range(pred_start, pred_start + len(self.predicted_series))
                pred_to_show = self.predicted_series[:len(pred_x)]

                if pred_start < len(self.all_counts) + FORECAST_STEPS:
                    ax_full.plot(pred_x, pred_to_show,
                                 'r--', label='Predicted', linewidth=2)

            ax_full.set_title('Full Timeline')
            ax_full.set_xlabel('Frame Number')
            ax_full.set_ylabel('Vehicle Count')
            ax_full.legend()
            ax_full.grid(True, alpha=0.3)

        # Short window plot (last 10 seconds)
        ax_short.clear()
        if len(self.timestamps) > 0 and len(self.all_counts) > 0:
            current_time = self.timestamps[-1]
            time_window = current_time - self.short_window_sec

            # Find indices within the last 10 seconds
            short_indices = [i for i, t in enumerate(self.timestamps) if t >= time_window]

            if short_indices:
                short_x = [self.timestamps[i] - time_window for i in short_indices]  # Relative time
                short_actual = [self.all_counts[i] for i in short_indices]

                # Plot actual counts
                ax_short.plot(short_x, short_actual,
                              'b-', label='Actual (10s)', linewidth=2)

                # Plot predictions if available
                if self.prediction_start_idx >= 0 and len(self.predicted_series) > 0:
                    pred_indices = [i for i in short_indices
                                    if i >= self.prediction_start_idx and
                                    i - self.prediction_start_idx < len(self.predicted_series)]

                    if pred_indices:
                        pred_x = [self.timestamps[i] - time_window for i in pred_indices]
                        pred_y = [self.predicted_series[i - self.prediction_start_idx]
                                  for i in pred_indices]
                        ax_short.plot(pred_x, pred_y,
                                      'r--', label='Predicted (10s)', linewidth=2)

                ax_short.set_title('Last 10 Seconds')
                ax_short.set_xlabel('Time (seconds)')
                ax_short.set_ylabel('Vehicle Count')
                ax_short.legend()
                ax_short.grid(True, alpha=0.3)
                ax_short.set_xlim(0, self.short_window_sec)

        # Auto-scale y-axes together
        if len(self.all_counts) > 0:
            y_min = min(self.all_counts[-self.max_display_points:])
            y_max = max(self.all_counts[-self.max_display_points:])

            if self.prediction_start_idx >= 0:
                valid_preds = [x for x in self.predicted_series if not np.isnan(x)]
                if valid_preds:
                    y_min = min(y_min, min(valid_preds))
                    y_max = max(y_max, max(valid_preds))

            padding = 0.1 * (y_max - y_min) if y_max != y_min else 1
            ax_full.set_ylim(max(0, y_min - padding), y_max + padding)
            ax_short.set_ylim(max(0, y_min - padding), y_max + padding)


if __name__ == "__main__":
    print("Starting Traffic Monitoring System...")
    print("Press 'm' to toggle metrics display")
    system = UnifiedTrafficSystem()
    system.run("traffic_prediction1.mp4")  # or use 0 for webcam