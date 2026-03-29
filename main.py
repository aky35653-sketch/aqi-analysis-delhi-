import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample AQI data
months = np.array([1, 2, 3, 4]).reshape(-1, 1)
aqi = np.array([320, 280, 250, 200])

# Train model
model = LinearRegression()
model.fit(months, aqi)

# Predict future AQI
prediction = model.predict([[5]])
print("Predicted AQI:", prediction[0])

# Plot graph
plt.plot(months, aqi, marker='o')
plt.title("AQI Trend")
plt.xlabel("Month")
plt.ylabel("AQI")
plt.show()
