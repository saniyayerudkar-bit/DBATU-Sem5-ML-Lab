import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Import libraries and create dataset
X, y = make_blobs(n_samples=120, centers=2, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("SVM Sample Dataset")
plt.show()

# 2. Train three SVM models
linear = SVC(kernel="linear", C=2)
linear.fit(X_train, y_train)

rbf = SVC(kernel="rbf", C=2, gamma=0.5)
rbf.fit(X_train, y_train)

poly = SVC(kernel="poly", degree=4, C=2)
poly.fit(X_train, y_train)

print("Linear Accuracy:", accuracy_score(y_test, linear.predict(X_test)))
print("RBF Accuracy:", accuracy_score(y_test, rbf.predict(X_test)))
print("Polynomial Accuracy:", accuracy_score(y_test, poly.predict(X_test)))

# 3. Visualize decision boundary and support vectors
model = rbf  # Change to linear or poly if desired

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 250),
    np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 250)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="blue")
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=180,
    facecolors="none",
    edgecolors="green",
    label="Support Vectors"
)

# Show margin lines for linear SVM
if model.kernel == "linear":
    decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    decision = decision.reshape(xx.shape)
    plt.contour(
        xx, yy, decision,
        levels=[-1, 0, 1],
        colors="black",
        linestyles=["--", "-", "--"]
    )

plt.title("SVM: Decision Boundary and Support Vectors")
plt.legend()
plt.show()

# 4. Try different C and gamma values
for c in [0.5, 2, 20]:
    model_c = SVC(kernel="rbf", C=c, gamma=0.5)
    model_c.fit(X_train, y_train)
    print("C =", c, "Accuracy =", accuracy_score(y_test, model_c.predict(X_test)))

for g in [0.5, 2, 5]:
    model_g = SVC(kernel="rbf", C=2, gamma=g)
    model_g.fit(X_train, y_train)
    print("Gamma =", g, "Accuracy =", accuracy_score(y_test, model_g.predict(X_test)))
