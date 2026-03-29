import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

base_path = 'Dataset/Time series dataset/Meteorological dataset/'

def load_and_prep(folder_name, col_name):
    # Find the 2023 file for each variable
    file_path = f"{base_path}{folder_name}/{folder_name}_2023.csv"
    df = pd.read_csv(file_path)

    # Standardize time and set it at the index for merging
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time')


    #Keep only the value column (not time column) and rename it
    return df[[df.columns[0]]].rename(columns={df.columns[0]: col_name})

#Load each variable seperately
irr = load_and_prep('Irradiance', 'Irradiance')
temp = load_and_prep('Temperature', 'Temp')
wind = load_and_prep('Wind', 'Wind_Speed')
rain = load_and_prep('Rainfall', 'Rainfall')
humidity = load_and_prep('Relative Humidity', 'RH')

# Making one big table for everything instead of seperate files
weather_2023 = pd.concat([irr, temp, wind, rain, humidity], axis = 1)

#Load the PV Inverter: SQ1
pv = pd.read_csv('Dataset/Time series dataset/PV generation dataset/PV stations with panel level optimizer/Inverter level dataset/SQ1_Inverter.csv')
pv['Time'] = pd.to_datetime(pv['Time'])
pv = pv.set_index('Time')

#Take the average of 1-min weather to 5 min intervals to match data to power output
weather_5min = weather_2023.resample('5min').mean()

#Merge everything
final_df = pd.merge(pv, weather_5min, left_index=True, right_index=True).dropna()

# Only do daylight (for now) so the zeros don't confuse the algorithm
final_df = final_df[final_df['Irradiance']> 10]

final_df['Irr_Temp_Interaction'] = final_df['Irradiance'] * final_df['Temp']

final_df['Wind_Speed_Sqrt'] = np.sqrt(final_df['Wind_Speed'])

print(final_df)

# --- MODEL TRAINING ---
# Define X and y
X = final_df[['Irradiance', 'Temp', 'Wind_Speed_Sqrt', 'Irr_Temp_Interaction', 'Rainfall', 'RH']]
y = final_df['totalActivePower(W)']

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and Metrics
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*30)
print(f"MODEL PERFORMANCE: SQ1")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.2f} Watts")
print("="*30)

coeffs = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Impact (Coefficients):")
print(coeffs)


# Graphing Relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Power Output (W)')
plt.ylabel('Predicted Power Output (W)')
plt.title('Actual vs. Predicted PV Power (SQ1)')
plt.show()

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).sort_index()
sample_days = comparison_df.loc['2023-07-01':'2023-07-07']

plt.figure(figsize=(15, 6))
plt.plot(sample_days.index, sample_days['Actual'], label='Actual Data', alpha=0.7)
plt.plot(sample_days.index, sample_days['Predicted'], label='Model Prediction', linestyle='--', alpha=0.9)
plt.legend()
plt.title('PV Generation: Actual vs Predicted (One Week Sample)')
plt.ylabel('Watts')
plt.show()



