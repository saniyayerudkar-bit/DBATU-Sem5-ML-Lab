import numpy as np
import matplotlib.pyplot as plt

# --- Part A: Single-Layer Perceptron from Scratch ---
# Simple linearly separable data (two input features)
X = np.array([
    [2, 2],
    [3, 2],
    [2, 3],
    [6, 6],
    [7, 6],
    [6, 7]
])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0
learning_rate = 0.2
epochs = 15

# Train the perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        total = np.dot(X[i], weights) + bias
        prediction = 1 if total >= 0 else 0
        error = y[i] - prediction
        weights = weights + learning_rate * error * X[i]
        bias = bias + learning_rate * error

print("Final weights:", weights)
print("Final bias:", bias)

# Test the Perceptron
predictions = []
for sample in X:
    total = np.dot(sample, weights) + bias
    prediction = 1 if total >= 0 else 0
    predictions.append(prediction)

print("Actual Output:   ", y)
print("Predicted Output:", predictions)

# --- Part B: Multilayer Perceptron using Scikit-learn ---
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_iris = iris.data[:100, 1:3]
y_iris = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.25, random_state=10
)

# Train MLP with Tanh activation
mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="tanh",
    max_iter=1200,
    random_state=10
)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Compare different activation functions
activation_functions = ["tanh", "relu", "logistic"]
results = []
for activation in activation_functions:
    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation=activation,
        max_iter=1200,
        random_state=10
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    results.append(accuracy)
    print(activation, "Accuracy:", round(accuracy * 100, 2), "%")

# Plot activation function comparison
plt.bar(activation_functions, results, color=["blue", "green", "orange"])
plt.xlabel("Activation Functions")
plt.ylabel("Accuracy Score")
plt.title("Comparison of MLP Activation Functions")
plt.ylim(0, 1.05)
plt.show()
