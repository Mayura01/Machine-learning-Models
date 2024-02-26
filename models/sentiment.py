import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import movie_reviews
import random

nltk.download('movie_reviews')


documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

X = [document for (document, label) in documents]
y = [label for (document, label) in documents]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_tfidf, y_train, epochs=10, validation_data=(X_test_tfidf, y_test))

test_loss, test_acc = model.evaluate(X_test_tfidf, y_test)
print('Neural Network Model Test accuracy:', test_acc)

predictions = model.predict(X_test_tfidf)
