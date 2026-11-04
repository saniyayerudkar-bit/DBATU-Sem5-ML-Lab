import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten

# 1 & 2. Load and display the digit dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

plt.imshow(X_train[0], cmap="gray")
plt.title("Sample Digit: " + str(y_train[0]))
plt.axis("off")
plt.show()

# 3. Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4. Build a simple DNN model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
model.summary()

# 5. Compile and train the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# 6. Visualize training and validation loss
plt.plot(history.history["loss"], marker="o", label="Training Loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# 7. Visualize training and validation accuracy
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# 8. Visualize hidden-layer activations
activation_model = Model(
    inputs=model.inputs,
    outputs=model.layers[1].output
)
activations = activation_model.predict(X_test[0:1])

plt.figure(figsize=(8, 3))
plt.bar(range(16), activations[0][:16])
plt.xlabel("Neuron Number")
plt.ylabel("Activation Value")
plt.title("First 16 Hidden-Layer Activations")
plt.show()

# 9. Predict a digit
prediction = model.predict(X_test[0:1])
predicted_digit = np.argmax(prediction)

plt.imshow(X_test[0], cmap="gray")
plt.title("Actual: " + str(y_test[0]) + " | Predicted: " + str(predicted_digit))
plt.axis("off")
plt.show()

print("Actual digit:", y_test[0])
print("Predicted digit:", predicted_digit)

# 10. Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
