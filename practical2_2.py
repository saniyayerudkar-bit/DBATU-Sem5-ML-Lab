# Step 1: Import Libraries
from sklearn.datasets import load_iris[cite: 38]
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold[cite: 38]
from sklearn.tree import DecisionTreeClassifier[cite: 38]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score[cite: 38]

# Step 2: Load the Iris Dataset
iris = load_iris()[cite: 38]
# Use the first 120 records
X = iris.data[:120][cite: 38]
y = iris.target[:120][cite: 38]
print("Total records:", len(X))[cite: 38]
print("Target classes:", iris.target_names)[cite: 38]

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10, stratify=y
)[cite: 38, 39]
model = DecisionTreeClassifier(max_depth=4, random_state=10)[cite: 39]
model.fit(X_train, y_train)[cite: 39]
y_pred = model.predict(X_test)[cite: 39]

print("Accuracy:", accuracy_score(y_test, y_pred))[cite: 39]
print("Precision:", precision_score(y_test, y_pred, average="macro"))[cite: 39]
print("Recall:", recall_score(y_test, y_pred, average="macro"))[cite: 39]
print("F1-score:", f1_score(y_test, y_pred, average="macro"))[cite: 39]

# Step 4: Apply 4-Fold Cross-Validation
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=10)[cite: 39]
scores = cross_validate(
    DecisionTreeClassifier(max_depth=4, random_state=10),
    X, y, cv=cv,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
)[cite: 39]

print("Average Accuracy:", scores["test_accuracy"].mean())[cite: 39]
print("Average Precision:", scores["test_precision_macro"].mean())[cite: 39]
print("Average Recall:", scores["test_recall_macro"].mean())[cite: 39]
print("Average F1-score:", scores["test_f1_macro"].mean())[cite: 39]

# Step 5: Compare the Results
print("TRAIN/TEST SPLIT RESULTS")[cite: 39, 40]
print("Accuracy:", accuracy_score(y_test, y_pred))[cite: 39, 40]
print("Precision:", precision_score(y_test, y_pred, average="macro"))[cite: 39, 40]
print("Recall :", recall_score(y_test, y_pred, average="macro"))[cite: 39, 40]
print("F1-score:", f1_score(y_test, y_pred, average="macro"))[cite: 39, 40]

print("\n4-FOLD CROSS-VALIDATION (AVERAGE)")[cite: 40]
print("Accuracy:", scores["test_accuracy"].mean())[cite: 40]
print("Precision:", scores["test_precision_macro"].mean())[cite: 40]
print("Recall :", scores["test_recall_macro"].mean())[cite: 40]
print("F1-score:", scores["test_f1_macro"].mean())[cite: 40]
