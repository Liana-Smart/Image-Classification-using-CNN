import tensorflow as tf
import matplotlib.pyplot as plt

#1. Load Data
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

#2. Visualize Data
#plt.imshow(X_train[0])
#plt.show()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add channel dimension
X_train = X_train[..., None]
X_test = X_test[..., None]

#3. Build Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(
        (2,2)
    ),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),
    
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs = 1
)

# Evaluation
loss, accuracy = model.evaluate(
    X_test,
    y_test
)
print(accuracy)

# Predictoin
prediction = model.predict(X_test[:1])
print(prediction.argmax())

labels = [
"T-Shirt",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Boot"
]

print(labels[prediction.argmax()])