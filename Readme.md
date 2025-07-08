*Predicting Future Climate Trends: Temperature Anomaly & Sea Level Rise using ARIMA and SARIMA*
---

Climate change is one of the most pressing challenges of our time, with rising global temperatures contributing to significant environmental shifts, including sea level rise. Understanding and predicting these trends is crucial for planning and mitigation efforts.

This project aims to develop a time-series forecasting model that predicts:

- Global Temperature Anomalies (°C) – Deviations from the historical average temperature, indicating climate warming.
  
- Sea Level Rise (mm) – The increase in global sea levels due to thermal expansion and melting ice caps.

Using historical climate data, we apply the ARIMA (AutoRegressive Integrated Moving Average) model and SARIMA (Seasonal AutoRegressive Integrated Moving Average), widely used statistical forecasting methods, to predict future values based on past trends.

-----

*Tech Stack*

- Python – Core programming language

- Pandas – Data manipulation and preprocessing
  
- NumPy – Numerical operations

- Matplotlib – Data visualization

- Statsmodels (ARIMA, SARIMA) – Time series forecasting

- scikit-learn – Evaluation using R² Score

- Jupyter Notebook / Google Colab – Interactive coding environment

----

*Features*

Loads and cleans historical global temperature anomaly and sea level data.

Merges datasets based on year for joint analysis.

Automatically searches for the best ARIMA and SARIMA model parameters to optimize prediction performance (based on R² score).

Forecasts future values for both:

Global Temperature Anomalies

Sea Level Rise

------

