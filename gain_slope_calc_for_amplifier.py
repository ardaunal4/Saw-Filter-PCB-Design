"""
Created on Wed Nov  3 09:18:26 2021
Description: Gain Slope Calculation For Low Noise Amplifier
@author: ardau
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

freq = [50, 100, 200, 400, 800, 1600, 3200]
gain = [24.45, 23.95, 23.45, 22.12, 19.08, 14.42, 9.23]

linear_reg = LinearRegression()
x = np.array(freq)
y = np.array(gain)
# fix shapes for regression
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
linear_reg.fit(x, y)
array = np.array(freq).reshape(-1, 1)
y_head = linear_reg.predict(array)

print("SLOPE : ", abs(y_head[1]-y_head[0]))

plt.figure(figsize = (10, 8))
plt.plot(freq, gain, color = "blue", label = "data")
plt.plot(array, y_head, color = "red", label = "LinearRegression")
plt.legend()
plt.grid("on")
plt.xlabel("FREQUENCY")
plt.ylabel("GAIN")
plt.title("GAIN SLOPE CALCULATION")
plt.show()  