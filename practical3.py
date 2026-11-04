import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# Part A: Simple Linear Regression using NumPy
# ==========================================

# Input data
X = np.array([2, 4, 6, 8, 10])[cite: 42]
Y = np.array([25, 35, 45, 55, 65])[cite: 42]

# Find slope and intercept
slope, intercept = np.polyfit(X, Y, 1)[cite: 42]

# Predict Y values
Y_pred = slope * X + intercept[cite: 42]

# Calculate RMSE
rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))[cite: 42]

# Calculate R-squared
ss_total = np.sum((Y - np.mean(Y)) ** 2)[cite: 42]
ss_error = np.sum((Y - Y_pred) ** 2)[cite: 42]
r2 = 1 - (ss_error / ss_total)[cite: 42]

print("Slope:", slope)[cite: 42]
print("Intercept:", intercept)[cite: 42]
print("Predicted values:", Y_pred)[cite: 42]
print("RMSE:", rmse)[cite: 42]
print("R-squared:", r2)[cite: 42]

# Plot regression line and residuals
plt.scatter(X, Y, color="green", label="Actual Data")[cite: 42]
plt.plot(X, Y_pred, color="orange", label="Regression Line")[cite: 42]

# Show residuals
for i in range(len(X)):
    plt.plot([X[i], X[i]], [Y[i], Y_pred[i]], "b--")[cite: 42]

plt.xlabel("Input X")[cite: 42]
plt.ylabel("Output Y")[cite: 42]
plt.title("Simple Linear Regression Model")[cite: 43]
plt.legend()[cite: 43]
plt.grid()[cite: 43]
plt.show()[cite: 43]


# ==========================================
# Part B: Multiple Linear Regression using Scikit-learn
# ==========================================

X_multi = np.array([
    [2, 55],
    [3, 60],
    [4, 65],
    [5, 70],
    [6, 75],
    [7, 80],
    [8, 85]
])[cite: 43]

Y_multi = np.array([42, 48, 54, 60, 68, 74, 82])[cite: 43]

model = LinearRegression()[cite: 43]
model.fit(X_multi, Y_multi)[cite: 43]

Y_pred_multi = model.predict(X_multi)[cite: 43]
rmse_multi = np.sqrt(mean_squared_error(Y_multi, Y_pred_multi))[cite: 43]
r2_multi = r2_score(Y_multi, Y_pred_multi)[cite: 43]

print("Intercept:", model.intercept_)[cite: 43]
print("Coefficients:", model.coef_)[cite: 43]
print("Predicted marks:", Y_pred_multi)[cite: 43]
print("RMSE:", rmse_multi)[cite: 43]
print("R-squared:", r2_multi)[cite: 43]

# Plot actual and predicted values
plt.scatter(Y_multi, Y_pred_multi, color="purple")[cite: 44]
plt.plot([min(Y_multi), max(Y_multi)], [min(Y_multi), max(Y_multi)], "g--")[cite: 44]
plt.xlabel("Actual Scores")[cite: 44]
plt.ylabel("Predicted Scores")
