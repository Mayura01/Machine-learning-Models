import tensorflow as tf
from tensorflow.keras import layers, models
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

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'dataset\\train')
dataset_test_dir = os.path.join(current_dir, 'dataset\\test')
classes = ['dogs', 'cats']
train_images = []
train_labels = []
test_images = []
test_labels = []
target_size = (28, 28)

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        resized_image = cv2.resize(image, target_size)
        train_images.append(resized_image.reshape((*target_size, 1)))  # Reshape to (28, 28, 1)
        train_labels.append(class_idx)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_test_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        resized_image = cv2.resize(image, target_size)
        test_images.append(resized_image.reshape((*target_size, 1)))  # Reshape to (28, 28, 1)
        test_labels.append(class_idx)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
