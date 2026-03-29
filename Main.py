import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading Data
base_path = 'Dataset/Time series dataset/Meteorological dataset/'

def load_and_prep(folder_name, col_name):
    file_path = f"{base_path}{folder_name}/{folder_name}_2023.csv"
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time')
    return df[[df.columns[0]]].rename(columns={df.columns[0]: col_name})

# Load variables
irr = load_and_prep('Irradiance', 'Irradiance')
temp = load_and_prep('Temperature', 'Temp')
wind = load_and_prep('Wind', 'Wind_Speed')
rain = load_and_prep('Rainfall', 'Rainfall')
humidity = load_and_prep('Relative Humidity', 'RH')

# Combine weather and resample to match 5-min PV data to match power output intervals.
weather_2023 = pd.concat([irr, temp, wind, rain, humidity], axis=1)
weather_5min = weather_2023.resample('5min').mean()

# Load PV Inverter Data
pv = pd.read_csv('Dataset/Time series dataset/PV generation dataset/PV stations with panel level optimizer/Inverter level dataset/SQ1_Inverter.csv')
pv['Time'] = pd.to_datetime(pv['Time'])
pv = pv.set_index('Time')

# Final Merge
final_df = pd.merge(pv, weather_5min, left_index=True, right_index=True).dropna()

# P is proportional to G * Temp-Efficiency)
final_df['Irr_Temp_Interaction'] = final_df['Irradiance'] * final_df['Temp']
# Non-linear wind cooling (Usually square root relationship)
final_df['Wind_Speed_Sqrt'] = np.sqrt(final_df['Wind_Speed'])

# 1. Daylight filter
# 2. Clipping filter (Hardware limit at ~27.5kW)
# 3. RRemove cases where sun is out but power is 0 due to maintenance/errors
filtered_df = final_df[
    (final_df['Irradiance'] > 10) &
    (final_df['totalActivePower(W)'] < 27000) &
    ~((final_df['totalActivePower(W)'] < 100) & (final_df['Irradiance'] > 100))
].copy()

# Model Training
features = ['Irradiance', 'Temp', 'Wind_Speed_Sqrt', 'Irr_Temp_Interaction', 'Rainfall', 'RH']
X = filtered_df[features]
y = filtered_df['totalActivePower(W)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MODEL PERFORMANCE: SQ1_Inverter (Filtered)\nR-squared: {r2:.4f}\nRMSE: {rmse:.2f} Watts\n")

coeffs = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nCoefficients:")
print(coeffs)

# Graphing
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs. Predicted PV Power (Filtered for Physics)')
plt.xlabel('Actual Power Output (W)')
plt.ylabel('Predicted Power Output (W)')
plt.show()

# Time Series Sample
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).sort_index()
sample_days = comparison_df.loc['2023-07-01':'2023-07-07']

plt.figure(figsize=(15, 6))
plt.plot(sample_days.index, sample_days['Actual'], label='Actual', alpha=0.7)
plt.plot(sample_days.index, sample_days['Predicted'], label='Model', linestyle='--', alpha=0.9)
plt.title('One Week Sample: Performance Tracking')
plt.legend()
plt.show()