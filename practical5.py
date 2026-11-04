# Import required libraries.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
# Use only two features for easy visualization
X = iris.data[:, 0:2]
y = iris.target
print("Dataset loaded successfully")
print("Number of samples:", len(X))
print("Features used:", iris.feature_names[0:2])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Predicted Classes:", y_pred)
print("Actual Classes:  ", y_test)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Visualize the Classification
plt.figure(figsize=(8, 6))
for class_no in np.unique(y_test):
    plt.scatter(
        X_test[y_test == class_no, 0],
        X_test[y_test == class_no, 1],
        label=iris.target_names[class_no]
    )
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("KNN Classification using Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()

# Compare Different Values of K
k_values = [2, 4, 6, 8, 10]
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = accuracy_score(y_test, prediction)
    accuracies.append(score)
    print("K =", k, "Accuracy =", round(score * 100, 2), "%")

# Plot accuracy for different K values
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker="s", color="green")
plt.xlabel("Value of K")
plt.ylabel("Classification Accuracy")
plt.title("KNN Accuracy for Various K Values")
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Effect of Number of Features (Curse of Dimensionality)
from sklearn.preprocessing import StandardScaler

# Use all four features
X_all = iris.data
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y, test_size=0.25, random_state=10
)

# Standardize all four features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test_all)

# Train KNN using all four features
knn_all = KNeighborsClassifier(n_neighbors=5)
knn_all.fit(X_train_scaled, y_train_all)
y_pred_all = knn_all.predict(X_test_scaled)
accuracy_all = accuracy_score(y_test_all, y_pred_all)

print("Accuracy using 2 features:", round(accuracy * 100, 2), "%")
print("Accuracy using 4 standardized features:", round(accuracy_all * 100, 2), "%")
