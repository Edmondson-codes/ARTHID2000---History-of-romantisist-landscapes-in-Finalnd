from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Activation, Dense, Flatten, Conv2D, MaxPooling2D
from keras.src.optimizers import Adam
# from keras.src.metrics import
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ================= Vars =================

BATCH_SIZE = 10

# ================= Get data =================
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape    # TODO: h & w here, reshape using them

X = lfw_people.data
n_features = X.shape[1]

X.reshape(n_samples, h, w, 3)

print(f"Shape: {X.shape}")

y = lfw_people.target
target_names = lfw_people.target_names

n_classes = target_names.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TODO: reformat X & y so that they have shape (n_samples, height, width, channels)
# currently: n_samples = 1288,

data_generator = ImageDataGenerator(horizontal_flip=True)

# error here
train_generator = data_generator.flow(X_train, y_train, BATCH_SIZE).shuffle()  # Added shuffle

# ================= Model =================
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
    Dense(units=32, activation='relu'),
    Dense(units=7, activation='softmax')
])


# ================= Training =================

model.compile(
    optimizer=Adam(learning_rate=0.0001), # learning rate
	loss='categorical_crossentropy',
	metrics=['accuracy','loss']
)

model.fit(
	train_generator,
	validation_split=0.05,
	epochs=10,
	verbose=2
)

# ================= Results =================

predictions = model.predict(x=X_test, batch_size=BATCH_SIZE, verbose=0)
rounded_predictions = np.argmax(predictions, axis=-1)
correct = rounded_predictions==y_test

print(f'Classification report: {classification_report(y_test, rounded_predictions)}')
