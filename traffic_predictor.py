import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from config import *

class TrafficPredictor:
    def __init__(self):
        self.classifier = None
        self.forecaster = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.forecast_scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, filepath):
        df = pd.read_csv(filepath)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.set_index('date_time')
        df['traffic_volume'] = df['traffic_volume'] / NORMALIZATION_FACTOR

        numeric_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
        df_numeric = df[numeric_cols].resample('H').mean().ffill()

        median_volume = df_numeric['traffic_volume'].median()
        df_numeric['is_traffic'] = (df_numeric['traffic_volume'] > median_volume).astype(int)

        features = df_numeric[['traffic_volume', 'temp', 'rain_1h']]
        scaled_data = self.scaler.fit_transform(features)

        X, y = [], []
        for i in range(SEQ_LENGTH, len(scaled_data)):
            X.append(scaled_data[i - SEQ_LENGTH:i])
            y.append(df_numeric['is_traffic'].iloc[i])

        return np.array(X), np.array(y)

    def build_classifier(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_forecaster(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(FORECAST_STEPS)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_classifier(self, filepath, epochs=10):
        X, y = self.prepare_data(filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        self.classifier = self.build_classifier((X_train.shape[1], X_train.shape[2]))
        history = self.classifier.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

        self.classifier.save(TRAFFIC_MODEL_PATH)
        joblib.dump(self.scaler, 'classifier_scaler.save')
        return history

    def train_forecaster(self, filepath, epochs=20):
        df = pd.read_csv(filepath)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.set_index('date_time')
        df['traffic_volume'] = df['traffic_volume'] / NORMALIZATION_FACTOR

        volumes = df['traffic_volume'].resample('H').mean().ffill().values.reshape(-1, 1)
        scaled_volumes = self.forecast_scaler.fit_transform(volumes)

        X, y = [], []
        for i in range(SEQ_LENGTH, len(scaled_volumes) - FORECAST_STEPS):
            X.append(scaled_volumes[i - SEQ_LENGTH:i])
            y.append(scaled_volumes[i:i + FORECAST_STEPS].flatten())

        X, y = np.array(X), np.array(y)

        self.forecaster = self.build_forecaster((X.shape[1], X.shape[2]))
        history = self.forecaster.fit(X, y, epochs=epochs, validation_split=0.2)

        self.forecaster.save(FORECAST_MODEL_PATH)
        joblib.dump(self.forecast_scaler, 'forecast_scaler.save')
        return history

    def load_models(self):
        self.classifier = load_model(TRAFFIC_MODEL_PATH)
        self.forecaster = load_model(FORECAST_MODEL_PATH)
        self.scaler = joblib.load('classifier_scaler.save')
        self.forecast_scaler = joblib.load('forecast_scaler.save')
        return self

    def predict_traffic(self, historical_counts):
        features = np.array([[historical_counts[-1], 20, 0]])  # Latest count, temp=20, rain=0
        scaled = self.scaler.transform(features)

        if len(scaled) < SEQ_LENGTH:
            padded = np.zeros((SEQ_LENGTH, scaled.shape[1]))
            padded[-len(scaled):] = scaled
            scaled = padded

        scaled = scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, -1)
        prediction = self.classifier.predict(scaled)[0][0]
        return prediction > THRESHOLD, prediction

    def predict_future(self, historical_counts):
        scaled = np.array(historical_counts).reshape(-1, 1) / NORMALIZATION_FACTOR
        scaled = self.forecast_scaler.transform(scaled)

        if len(scaled) < SEQ_LENGTH:
            padded = np.zeros((SEQ_LENGTH, 1))
            padded[-len(scaled):] = scaled
            scaled = padded

        scaled = scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
        predictions = self.forecaster.predict(scaled)[0]
        predictions = self.forecast_scaler.inverse_transform(predictions.reshape(-1, 1))
        return predictions.flatten() * NORMALIZATION_FACTOR




# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from config import SEQ_LENGTH, THRESHOLD, TRAFFIC_MODEL_PATH
# import joblib
#
#
# class TrafficPredictor:
#     def __init__(self):
#         self.model = None
#         self.scaler = MinMaxScaler(feature_range=(0, 1))
#
#     def prepare_data(self, filepath):
#         # Load and preprocess data
#         df = pd.read_csv(filepath)
#
#         # Convert date_time to datetime and set as index
#         df['date_time'] = pd.to_datetime(df['date_time'])
#         df = df.set_index('date_time')
#
#         # Select only numeric columns for resampling
#         numeric_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
#         df_numeric = df[numeric_cols].copy()
#
#         # Normalize traffic_volume by dividing by 325
#         df_numeric['traffic_volume'] = df_numeric['traffic_volume'] / 325
#
#         # Resample hourly and fill missing values
#         df_resampled = df_numeric.resample('H').mean().ffill()
#
#         # Create target variable
#         median_volume = df_resampled['traffic_volume'].median()
#         df_resampled['is_traffic'] = (df_resampled['traffic_volume'] > median_volume).astype(int)
#
#         # Feature selection
#         features = df_resampled[['traffic_volume', 'temp', 'rain_1h']]
#
#         # Scale data
#         scaled_data = self.scaler.fit_transform(features)
#
#         # Create sequences
#         X, y = [], []
#         for i in range(SEQ_LENGTH, len(scaled_data)):
#             X.append(scaled_data[i - SEQ_LENGTH:i])
#             y.append(df_resampled['is_traffic'].iloc[i])
#
#         return np.array(X), np.array(y)
#
#     def build_model(self, input_shape):
#         model = Sequential([
#             LSTM(64, return_sequences=True, input_shape=input_shape),
#             Dropout(0.2),
#             LSTM(32),
#             Dropout(0.2),
#             Dense(1, activation='sigmoid')
#         ])
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         return model
#
#     def train(self, filepath, epochs=10):
#         X, y = self.prepare_data(filepath)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
#         self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
#         history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
#
#         # Save model and scaler
#         self.model.save(TRAFFIC_MODEL_PATH)
#         joblib.dump(self.scaler, 'scaler.save')
#
#         return history
#
#     def predict(self, historical_counts):
#         # Scale input
#         scaled_data = self.scaler.transform(historical_counts)
#
#         # Reshape for LSTM
#         if len(scaled_data) < SEQ_LENGTH:
#             padded = np.zeros((SEQ_LENGTH, scaled_data.shape[1]))
#             padded[-len(scaled_data):] = scaled_data
#             scaled_data = padded
#
#         scaled_data = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, -1)
#
#         # Make prediction
#         prediction = self.model.predict(scaled_data)[0][0]
#         return prediction > THRESHOLD, prediction