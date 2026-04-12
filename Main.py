import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor

# --- DATA LOADING ---
base_path = 'Dataset/Time series dataset/Meteorological dataset/'


def load_and_prep(folder_name, col_name):
    file_path = f"{base_path}{folder_name}/{folder_name}_2023.csv"
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time')
    return df[[df.columns[0]]].rename(columns={df.columns[0]: col_name})


# Existing meteorological variables
irr = load_and_prep('Irradiance', 'Irradiance')
temp = load_and_prep('Temperature', 'Temp')
wind = load_and_prep('Wind', 'Wind_Speed')
rain = load_and_prep('Rainfall', 'Rainfall')
humidity = load_and_prep('Relative Humidity', 'RH')
pressure = load_and_prep('Sea Level Pressure', 'SLP')
visibility = load_and_prep('Visibility', 'Vis')

# Combine and resample weather to 5-min intervals
weather_2023 = pd.concat([irr, temp, wind, rain, humidity, pressure, visibility], axis=1)
weather_5min = weather_2023.resample('5min').mean()

# Load PV Inverter Data (SQ1)
pv = pd.read_csv(
    'Dataset/Time series dataset/PV generation dataset/PV stations with panel level optimizer/Inverter level dataset/SQ1_Inverter.csv')
pv['Time'] = pd.to_datetime(pv['Time'])
pv = pv.set_index('Time')

# Merge PV and Weather
final_df = pd.merge(pv, weather_5min, left_index=True, right_index=True).dropna()

# --- FEATURE ENGINEERING ---
# 1. Existing transformations
final_df['Wind_Speed_Sqrt'] = np.sqrt(final_df['Wind_Speed'])

# 2. TRUE Physics-Informed Feature: Estimated Cell Temperature (T_cell)
# Assuming standard NOCT of 45C. Estimates the actual temperature of the solar cell.
NOCT = 45
final_df['Est_Cell_Temp'] = final_df['Temp'] + ((NOCT - 20) / 800) * final_df['Irradiance']

# 3. Physics-Informed Interaction: Irradiance modified by Cell Temperature
# PV efficiency drops as cell temp rises. This captures thermodynamic degradation.
final_df['Physics_Irr_Temp_Interaction'] = final_df['Irradiance'] / (final_df['Est_Cell_Temp'] + 273.15)  # Using Kelvin

# --- PHYSICS-BASED FILTERING ---
filtered_df = final_df[
    (final_df['Irradiance'] > 10) &
    (final_df['totalActivePower(W)'] < 27000) &
    ~((final_df['totalActivePower(W)'] < 100) & (final_df['Irradiance'] > 100))
    ].copy()

# --- MODEL PREPARATION ---
features = [
    'Irradiance', 'Temp', 'Wind_Speed_Sqrt', 'Est_Cell_Temp',
    'Physics_Irr_Temp_Interaction', 'Rainfall', 'RH', 'SLP', 'Vis'
]
X = filtered_df[features]
y = filtered_df['totalActivePower(W)']

# CRITICAL: shuffle=False prevents time-series data leakage!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale X for ALL models to ensure fair comparison and faster convergence
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# --- MODEL COMPARISON ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),

    # CRITICAL: TransformedTargetRegressor scales 'y' for SVR, then inverse-scales the output automatically
    "SVR (RBF Kernel)": TransformedTargetRegressor(
        regressor=SVR(kernel='rbf', cache_size=1000),
        transformer=StandardScaler()
    )
}

results = {}

print(f"{'=' * 75}\nMulti-Model Performance Comparison (Chronological Split + Physics Features)\n{'=' * 75}")

for name, model in models.items():
    print(f"Training {name}...")

    # Feed scaled X to all models.
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = y_pred
    print(f"{name:20} | R2: {r2:.4f} | RMSE: {rmse:.2f} W | MAE: {mae:.2f} W")

# --- ANALYSIS & PLOTTING ---

# 1. Feature Importance for Random Forest
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nRandom Forest Feature Importances:")
print(importances)

# 2. Visual Comparison: Actual vs Predicted (Random Forest)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=results["Random Forest"], alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Random Forest: Actual vs. Predicted (Physics-Informed Features)')
plt.xlabel('Actual Power Output (W)')
plt.ylabel('Predicted Power Output (W)')
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(12, 10))  # Slightly larger to accommodate new features
sns.heatmap(filtered_df[features + ['totalActivePower(W)']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Environmental & Physics-Informed Factors vs Power')
plt.show()