# 1. Import libraries and create data
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Create simple synthetic data
X, y = make_moons(n_samples=300, noise=0.35, random_state=10)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

# 2. Display the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.xlabel("Attribute 1")
plt.ylabel("Attribute 2")
plt.title("Moon-Shaped Synthetic Dataset")
plt.show()

# 3. Decision Tree without pruning
model1 = DecisionTreeClassifier(random_state=10)
model1.fit(X_train, y_train)

# Calculate accuracy
print("Training Accuracy:", accuracy_score(y_train, model1.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, model1.predict(X_test)))

# Display decision tree
plt.figure(figsize=(12, 6))
plot_tree(model1, filled=True)
plt.show()

# 4. Decision Tree with depth control
# Limit tree depth to 3
model2 = DecisionTreeClassifier(max_depth=3, random_state=10)
model2.fit(X_train, y_train)
print("Training Accuracy:", accuracy_score(y_train, model2.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, model2.predict(X_test)))
plt.figure(figsize=(8, 5))
plot_tree(model2, filled=True)
plt.show()

# 5. Decision Tree with pruning
# Apply pruning
model3 = DecisionTreeClassifier(ccp_alpha=0.02, random_state=10)
model3.fit(X_train, y_train)
print("Training Accuracy:", accuracy_score(y_train, model3.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, model3.predict(X_test)))
plt.figure(figsize=(8, 5))
plot_tree(model3, filled=True)
plt.show()

# 6. Compare the three models
print("Model                      Training Accuracy    Testing Accuracy")
print("Without pruning           ", round(accuracy_score(y_train, model1.predict(X_train)), 2), "              ", round(accuracy_score(y_test, model1.predict(X_test)), 2))
print("Depth control             ", round(accuracy_score(y_train, model2.predict(X_train)), 2), "              ", round(accuracy_score(y_test, model2.predict(X_test)), 2))
print("With pruning              ", round(accuracy_score(y_train, model3.predict(X_train)), 2), "              ", round(accuracy_score(y_test, model3.predict(X_test)), 2))
