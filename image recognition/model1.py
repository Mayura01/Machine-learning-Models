import tensorflow as tf

#loading dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train, x_test = x_train / 255.0, x_test / 255.0

#stacking layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

#training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)


model.evaluate(x_test, y_test)


#for testing with example
import cv2
import numpy as np

# Loading and preprocessing input image
input_image = cv2.imread('5.png', cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(input_image, (28, 28))
normalized_image = resized_image / 255.0
input_data = np.expand_dims(normalized_image, axis=0)

predictions = model.predict(input_data)

predicted_digit = np.argmax(predictions)

print("Predicted Digit:", predicted_digit)

predictions = model.predict(x_test)