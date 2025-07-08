import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score

# Load Temperature Data
temp_df = pd.read_csv("/content/Cleaned_Global_Temperature.csv")
temp_df.columns = ["Year", "Temperature_Anomaly"]

# Load Sea Level Data
sea_df = pd.read_csv("/content/sealevel_cleaned.csv")
sea_df.columns = ["Year", "Sea_Level"]
sea_df = sea_df.groupby("Year").mean().reset_index()  # Ensure one row per year

# Merge the datasets
merged_df = pd.merge(temp_df, sea_df, on="Year", how="inner")

# Extract variables
years = merged_df["Year"].values
temperature_anomaly = merged_df["Temperature_Anomaly"].values
sea_level = merged_df["Sea_Level"].values

# Optimize ARIMA model for Temperature
best_temp_r2 = -1
best_temp_order = None
for p in range(3, 8):
    for q in range(3, 8):
        try:
            temp_model = ARIMA(temperature_anomaly, order=(p, 1, q)).fit()
            temp_pred = temp_model.fittedvalues
            r2_temp = r2_score(temperature_anomaly, temp_pred)
            if r2_temp > best_temp_r2:
                best_temp_r2 = r2_temp
                best_temp_order = (p, 1, q)
        except:
            continue

# Fit final temperature model with best order
final_temp_model = ARIMA(temperature_anomaly, order=best_temp_order).fit()
temp_pred = final_temp_model.fittedvalues
future_years = np.arange(2025, 2031)
future_temp_pred = final_temp_model.forecast(steps=len(future_years))

# Fit ARIMA model for Sea Level
sea_model = ARIMA(sea_level, order=(3, 1, 2)).fit()
sea_pred = sea_model.fittedvalues
future_sea_pred = sea_model.forecast(steps=len(future_years))

# Evaluate performance
r2_temp_final = r2_score(temperature_anomaly, temp_pred)
r2_sea_final = r2_score(sea_level, sea_pred)

# Correlation analysis
correlation = np.corrcoef(temperature_anomaly, sea_level)[0, 1]

print(f"Optimized R² Score (Temperature): {r2_temp_final:.4f}")
print(f"R² Score (Sea Level): {r2_sea_final:.4f}")
print(f"Best ARIMA Order for Temperature: {best_temp_order}")
print(f"Correlation between Temperature Anomaly and Sea Level: {correlation:.4f}")

# Plot results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Temperature Anomaly
ax1.set_xlabel("Year")
ax1.set_ylabel("Temperature Anomaly (°C)", color="red")
ax1.scatter(years, temperature_anomaly, color="red", label="Actual Temp Anomaly", alpha=0.6)
ax1.plot(years, temp_pred, color="darkred", linewidth=2, label="Predicted Temp Anomaly")
ax1.scatter(future_years, future_temp_pred, color="orange", edgecolors="black", s=80, label="Future Temp Predictions")

# Sea Level
ax2 = ax1.twinx()
ax2.set_ylabel("Global Mean Sea Level (mm)", color="blue")
ax2.scatter(years, sea_level, color="blue", label="Actual Sea Level", alpha=0.6)
ax2.plot(years, sea_pred, color="navy", linewidth=2, label="Predicted Sea Level")
ax2.scatter(future_years, future_sea_pred, color="cyan", edgecolors="black", s=80, label="Future Sea Level Predictions")

# Legends
ax1.legend(loc="upper left", bbox_to_anchor=(0, 1.1))
ax2.legend(loc="upper right", bbox_to_anchor=(1, 1.1))

plt.title("Temperature Anomalies vs Sea Level Rise (1950-2030)")
plt.grid(True)
plt.show()

# Simple Prediction Function
def predict_values(year):
    years_range = np.arange(years[0], year + 1)
    steps_ahead = len(years_range) - len(years)
    if steps_ahead <= 0:
        return temp_pred[years_range[-1] - years[0]], sea_pred[years_range[-1] - years[0]]
    temp_future = final_temp_model.forecast(steps=steps_ahead)[-1]
    sea_future = sea_model.forecast(steps=steps_ahead)[-1]
    return temp_future, sea_future

# User Input Loop
while True:
    year_input = input("Enter a year to predict (or type 'exit' to quit): ")
    if year_input.lower() == "exit":
        print("Exiting program.")
        break
    try:
        year_input = int(year_input)
        temp_anomaly, sea_level_pred = predict_values(year_input)
        print(f"Year: {year_input}")
        print(f"Predicted Temperature Anomaly: {temp_anomaly:.2f}°C")
        print(f"Predicted Sea Level Rise: {sea_level_pred:.2f} mm")
    except ValueError:
        print("Invalid input! Please enter a valid year or 'exit'.")
