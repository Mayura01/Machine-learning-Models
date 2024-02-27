import tensorflow as tf
from keras import layers, models
import os
import cv2
import numpy as np

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'dataset\\train')
dataset_test_dir = os.path.join(current_dir, 'dataset\\test')

classes = ['dogs', 'cats']
train_images = []
train_labels = []
test_images = []
test_labels = []

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (28, 28))
        train_images.append(resized_image.reshape((*resized_image.shape, 1)))
        train_labels.append(class_idx)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_test_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (28, 28))
        test_images.append(resized_image.reshape((*resized_image.shape, 1)))
        test_labels.append(class_idx)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
predictions = model.predict(test_images)

# Example: Predicting a single image
input_image = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(input_image, (28, 28))
normalized_image = resized_image / 255.0
input_data = np.expand_dims(normalized_image, axis=0)

predictions = model.predict(input_data)
predicted_animal = 'dog' if predictions[0][0] > 0.5 else 'cat'
print("Predicted animal:", predicted_animal)
