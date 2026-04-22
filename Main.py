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

# Loading the Data
base_path = 'Dataset/Time series dataset/Meteorological dataset/'


def load_and_prep(folder_name, col_name):
    file_path = f"{base_path}{folder_name}/{folder_name}_2023.csv"
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time')
    return df[[df.columns[0]]].rename(columns={df.columns[0]: col_name})


# The meteorological variables
irr = load_and_prep('Irradiance', 'Irradiance')
temp = load_and_prep('Temperature', 'Temp')
wind = load_and_prep('Wind', 'Wind_Speed')
rain = load_and_prep('Rainfall', 'Rainfall')
humidity = load_and_prep('Relative Humidity', 'RH')
pressure = load_and_prep('Sea Level Pressure', 'SLP')
visibility = load_and_prep('Visibility', 'Vis')

# Resample weather to 5-min intervals, since power is in 5 minute intervals
weather_2023 = pd.concat([irr, temp, wind, rain, humidity, pressure, visibility], axis=1)
weather_5min = weather_2023.resample('5min').mean()

# Load PV Inverter Data (SQ1)
pv = pd.read_csv(
    'Dataset/Time series dataset/PV generation dataset/PV stations with panel level optimizer/Inverter level dataset/SQ1_Inverter.csv')
pv['Time'] = pd.to_datetime(pv['Time'])
pv = pv.set_index('Time')

final_df = pd.merge(pv, weather_5min, left_index=True, right_index=True).dropna()

# Physics Based Alterations
final_df['Wind_Speed_Sqrt'] = np.sqrt(final_df['Wind_Speed'])

# Estimates the actual temperature of the solar cell, assuming a standard NOCT of 45C.
NOCT = 45
final_df['Est_Cell_Temp'] = final_df['Temp'] + ((NOCT - 20) / 800) * final_df['Irradiance']

final_df['Physics_Irr_Temp_Interaction'] = final_df['Irradiance'] / (final_df['Est_Cell_Temp'] + 273.15)  # Using Kelvin

# Filtering based on physics variables
filtered_df = final_df[
    (final_df['Irradiance'] > 10) &
    (final_df['totalActivePower(W)'] < 27000) &
    ~((final_df['totalActivePower(W)'] < 100) & (final_df['Irradiance'] > 100))
    ].copy()

# Model
features = [
    'Irradiance', 'Temp', 'Wind_Speed_Sqrt', 'Est_Cell_Temp',
    'Physics_Irr_Temp_Interaction', 'Rainfall', 'RH', 'SLP', 'Vis'
]
X = filtered_df[features]
y = filtered_df['totalActivePower(W)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale X for ALL models to ensure fair comparison
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "SVR (RBF Kernel)": TransformedTargetRegressor(
        regressor=SVR(kernel='rbf', cache_size=1000),
        transformer=StandardScaler()
    )
}

results = {}

print(f"{'=' * 75}\nMulti-Model Performance Comparison (Chronological Split + Physics Features)\n{'=' * 75}")

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = y_pred
    print(f"{name:20} | R2: {r2:.4f} | RMSE: {rmse:.2f} W | MAE: {mae:.2f} W")

# Graphing

rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nRandom Forest Feature Importances:")
print(importances)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=results["Random Forest"], alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Random Forest: Actual vs. Predicted (Physics-Informed Features)')
plt.xlabel('Actual Power Output (W)')
plt.ylabel('Predicted Power Output (W)')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(filtered_df[features + ['totalActivePower(W)']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Environmental & Physics-Informed Factors vs Power')
plt.show()

lr_model = models["Linear Regression"]

coeff_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_
})

coeff_df['Abs_Influence'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='Abs_Influence', ascending=False)

print(f"\n{'=' * 30}\nLinear Regression Coefficients\n{'=' * 30}")
print(coeff_df[['Feature', 'Coefficient']])

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coeff_df, palette='coolwarm')
plt.axvline(0, color='black', lw=1)
plt.title('Influence of Environmental Factors on Power Output (Linear Regression)')
plt.xlabel('Coefficient Value (Direction and Strength of Influence)')
plt.show()