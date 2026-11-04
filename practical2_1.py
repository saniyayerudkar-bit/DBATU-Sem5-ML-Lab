
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a Toy Dataset

X = np.array([
    [2, 1], [3, 1], [2, 2], [3, 2],
    [6, 5], [7, 5], [6, 6], [7, 6]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

print("Input data:")
print(X)

print("Actual classes:")
print(y)


# Step 2: Define Decision Boundary Manually

y_pred = []

for point in X:
    if point[0] + point[1] >= 8:
        y_pred.append(1)
    else:
        y_pred.append(0)

print("Predicted classes:")
print(y_pred)


# Step 3: Calculate Accuracy

correct = 0

for i in range(len(y)):
    if y[i] == y_pred[i]:
        correct += 1

accuracy = (correct / len(y)) * 100

print("Number of Correct Predictions:", correct)
print("Classification Accuracy:", accuracy, "%")


# Step 4: Visualize Dataset and Decision Boundary

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="blue")

x_line = np.array([0, 8])
y_line = 8 - x_line

plt.plot(x_line, y_line, "g--", label="X1 + X2 = 8")

plt.xlabel("Input Feature X1")
plt.ylabel("Input Feature X2")
plt.title("Manual Decision Boundary for Toy Classification")
plt.legend()
plt.grid()
plt.show()


# Step 5: Test a New Data Point

new_point = [4, 5]

if new_point[0] + new_point[1] >= 8:
    print("New point belongs to Class 1")
else:
    print("New point belongs to Class 0")
