# TIAA MF Outflows Forecasting

This project forecasts mutual fund (MF) outflows using advanced machine learning techniques. The project employs two complementary modeling approaches:

1. **LSTM (Long Short-Term Memory)** for time-series forecasting.
2. **Random Forest Regression** for feature-based prediction enhanced with PCA for dimensionality reduction.

## Project Overview

- **Data Simulation & Preprocessing:**  
  Synthetic MF outflow data is generated with trend, seasonality, and noise. Feature engineering includes lag features, rolling statistics, and calendar-based features.

- **LSTM Model:**  
  The LSTM model uses sliding windows on scaled data to capture temporal dependencies.

- **Random Forest Model:**  
  A Random Forest Regressor is applied on standardized features reduced via PCA to capture influential predictors.

- **Visualization & Evaluation:**  
  Both models are evaluated using RMSE, MAE, and R². Predictions are visualized against actual data.

## Files

- `main.py`: Contains the complete project code.
- `requirements.txt`: Lists all required Python packages.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Project overview and documentation.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/TIAA_MF_Outflows_Forecasting.git
   cd TIAA_MF_Outflows_Forecasting
