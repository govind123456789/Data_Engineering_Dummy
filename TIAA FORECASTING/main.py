
---

### 4. **main.py**

Create a file named **main.py** and paste the complete Python project code below. This code includes data generation, feature engineering, LSTM modeling, Random Forest modeling, evaluation, and visualization.

```python
"""
TIAA MF Outflows Forecasting
----------------------------
This script forecasts mutual fund outflows using two modeling approaches:
1. LSTM for time-series forecasting.
2. Random Forest Regressor for feature-based regression enhanced with PCA.

The project includes data simulation, preprocessing, model training, evaluation,
and visualization.

Run the script with: python main.py
"""

# ---------------------------
# 1. Import Required Libraries
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 2. Data Generation & Visualization
# ---------------------------
# Generate synthetic MF outflows data with trend, seasonality, and noise.
date_rng = pd.date_range(start='1/1/2015', end='12/31/2019', freq='D')
n = len(date_rng)

trend = np.linspace(0, 50, n)
seasonality = 10 * np.sin(np.linspace(0, 20 * np.pi, n))
noise = np.random.normal(0, 5, n)
mf_outflows = 100 + trend + seasonality + noise

# Create a DataFrame
df = pd.DataFrame({'Date': date_rng, 'MF_Outflows': mf_outflows})
df.set_index('Date', inplace=True)

# Visualize the synthetic time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['MF_Outflows'], label='MF Outflows', color='navy')
plt.title('Synthetic MF Outflows Time Series')
plt.xlabel('Date')
plt.ylabel('Outflows')
plt.legend()
plt.show()

# ---------------------------
# 3. Feature Engineering for Regression Model
# ---------------------------
df['Lag_1']          = df['MF_Outflows'].shift(1)
df['Lag_7']          = df['MF_Outflows'].shift(7)
df['Lag_30']         = df['MF_Outflows'].shift(30)
df['Rolling_Mean_7'] = df['MF_Outflows'].rolling(window=7).mean()
df['Rolling_Std_7']  = df['MF_Outflows'].rolling(window=7).std()
df['Month']          = df.index.month
df['DayOfWeek']      = df.index.dayofweek

# Drop initial rows with NaN values
df = df.dropna()
print("Data sample with engineered features:")
print(df.head())

# ---------------------------
# 4. Train-Test Split (Time-Based)
# ---------------------------
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df  = df.iloc[train_size:]

# ---------------------------
# 5. Time-Series Forecasting with LSTM
# ---------------------------
# Scale data for LSTM using MinMaxScaler.
lstm_scaler = MinMaxScaler()
train_scaled = lstm_scaler.fit_transform(train_df[['MF_Outflows']])
test_scaled  = lstm_scaler.transform(test_df[['MF_Outflows']])

# Function to create sequences for LSTM
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

window_size = 30
X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test   = create_sequences(test_scaled, window_size)
print("LSTM Training Data Shape:", X_train.shape, y_train.shape)

# Build and compile the LSTM model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

# Train the LSTM model
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                         validation_split=0.1, verbose=1)

# Plot LSTM training loss
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Make predictions with LSTM
predictions_lstm = lstm_model.predict(X_test)
predictions_lstm = lstm_scaler.inverse_transform(predictions_lstm)
y_test_actual   = lstm_scaler.inverse_transform(y_test)

# Evaluate LSTM performance
rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, predictions_lstm))
mae_lstm  = mean_absolute_error(y_test_actual, predictions_lstm)
r2_lstm   = r2_score(y_test_actual, predictions_lstm)
print("LSTM Model Performance:")
print("RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(rmse_lstm, mae_lstm, r2_lstm))

# Visualize LSTM predictions vs actual values
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual MF Outflows', color='blue')
plt.plot(predictions_lstm, label='LSTM Predictions', color='red', alpha=0.7)
plt.title('LSTM Predictions vs Actual MF Outflows')
plt.xlabel('Time Step')
plt.ylabel('MF Outflows')
plt.legend()
plt.show()

# ---------------------------
# 6. Feature-Based Regression with Random Forest & PCA
# ---------------------------
# Define features and target variable
features = ['Lag_1', 'Lag_7', 'Lag_30', 'Rolling_Mean_7', 'Rolling_Std_7', 'Month', 'DayOfWeek']
target   = 'MF_Outflows'

X_train_rf = train_df[features]
y_train_rf = train_df[target]
X_test_rf  = test_df[features]
y_test_rf  = test_df[target]

# Standardize the features
rf_scaler = StandardScaler()
X_train_rf_scaled = rf_scaler.fit_transform(X_train_rf)
X_test_rf_scaled  = rf_scaler.transform(X_test_rf)

# Apply PCA for dimensionality reduction (reducing to 5 components)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_rf_scaled)
X_test_pca  = pca.transform(X_test_rf_scaled)
print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)

# Build and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_train_rf)

# Make predictions with Random Forest
predictions_rf = rf_model.predict(X_test_pca)

# Evaluate Random Forest performance
rmse_rf = np.sqrt(mean_squared_error(y_test_rf, predictions_rf))
mae_rf  = mean_absolute_error(y_test_rf, predictions_rf)
r2_rf   = r2_score(y_test_rf, predictions_rf)
print("\nRandom Forest Model Performance:")
print("RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(rmse_rf, mae_rf, r2_rf))

# Visualize Random Forest predictions vs actual values
plt.figure(figsize=(12,6))
plt.plot(y_test_rf.values, label='Actual MF Outflows', color='green')
plt.plot(predictions_rf, label='Random Forest Predictions', color='orange', alpha=0.7)
plt.title('Random Forest Predictions vs Actual MF Outflows')
plt.xlabel('Time Step')
plt.ylabel('MF Outflows')
plt.legend()
plt.show()

# ---------------------------
# 7. Summary of Model Performance
# ---------------------------
print("\n--- Model Performance Summary ---")
print("LSTM Model --> RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(rmse_lstm, mae_lstm, r2_lstm))
print("Random Forest Model --> RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(rmse_rf, mae_rf, r2_rf))
