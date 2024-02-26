import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np

train_dir = 'D:\machine learning\Machine-learning-Models\image recognition\dataset'
test_dir = 'D:\machine learning\Machine-learning-Models\image recognition\dataset'

train_images = []
train_labels = []
test_images = []
test_labels = []

def read_and_preprocess_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(directory, filename))
            image = cv2.resize(image, (128, 128))
            images.append(image)
            label = filename.split('_')[0]
            labels.append(label)
    return np.array(images), np.array(labels)

train_images, train_labels = read_and_preprocess_images(train_dir)

test_images, test_labels = read_and_preprocess_images(test_dir)

print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
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

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
