from traffic_predictor import TrafficPredictor

if __name__ == "__main__":
    print("Training traffic prediction models...")
    predictor = TrafficPredictor()

    print("\nTraining classifier model...")
    predictor.train_classifier("training_data.csv", epochs=15)

    print("\nTraining forecasting model...")
    predictor.train_forecaster("training_data.csv", epochs=20)

    print("\nBoth models trained and saved successfully!")

# from traffic_predictor import TrafficPredictor
# predictor = TrafficPredictor()
# predictor.train("data.csv")