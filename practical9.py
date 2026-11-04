import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc

# 1. Load and prepare the data
url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
df = pd.read_csv(url)

cols = ["Glucose", "BloodPressure", "BMI"]
df[cols] = df[cols].replace(0, np.nan).fillna(df[cols].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
print(df.head())

# 2. Train the Logistic Regression model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=200, random_state=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 3. Sigmoid function
z = np.linspace(-12, 12, 120)
plt.plot(z, 1 / (1 + np.exp(-z)), color="green")
plt.title("Sigmoid Activation Function")
plt.xlabel("Input Value")
plt.ylabel("Output Probability")
plt.grid()
plt.show()

# 4. Decision boundary (Using Glucose and BloodPressure)
X2 = df[["Glucose", "BloodPressure"]]
a, b = np.meshgrid(
    np.arange(X2.Glucose.min(), X2.Glucose.max(), 3),
    np.arange(X2.BloodPressure.min(), X2.BloodPressure.max(), 2)
)
sc2 = StandardScaler()
X2_scaled = sc2.fit_transform(X2)
m2 = LogisticRegression(max_iter=200, random_state=10).fit(X2_scaled, y)

grid = sc2.transform(np.c_[a.ravel(), b.ravel()])
pred = m2.predict(grid).reshape(a.shape)

plt.contourf(a, b, pred, alpha=0.3)
plt.scatter(X2.Glucose, X2.BloodPressure, c=y, edgecolors="k")
plt.xlabel("Glucose")
plt.ylabel("Blood Pressure")
plt.title("Logistic Regression Decision Boundary")
plt.show()

# 5. ROC curve and AUC
prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, prob)
score = auc(fpr, tpr)

plt.plot(fpr, tpr, color="blue", label="AUC = %.3f" % score)
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")
plt.legend()
plt.grid()
plt.show()
print("AUC Score:", round(score, 3))
