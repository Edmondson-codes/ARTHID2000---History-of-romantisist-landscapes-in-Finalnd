from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical
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
NUM_CLASSES = 7

# ================= Get data =================
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape    # TODO: h & w here, reshape using them

X = lfw_people.data
n_features = X.shape[1]

X = X.reshape(n_samples, h, w, 1)

print(f"Shape of X: {X.shape}")                     # Shape of X: (1288, 50, 37, 1)
print(f"Shape of imgs: {lfw_people.images.shape}")  # Shape of imgs: (1288, 50, 37)

y = lfw_people.target
# y = to_categorical(y, num_classes=NUM_CLASSES)
target_names = lfw_people.target_names

print(f"Shape of y: {y.shape}")   # Shape of y: (1288,)

n_classes = target_names.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"labels Head: {y_train}")

data_generator = ImageDataGenerator()  # rescale=1./255

# error here
train_generator = data_generator.flow(X_train, y_train, BATCH_SIZE)

print(train_generator)


# ================= Model =================

model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
    Dense(units=32, activation='relu'),
    Flatten(),
    Dense(units=n_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
	loss='sparse_categorical_crossentropy',   # this makes the output a one-hot encoding
	metrics=['accuracy'],
)

print(model.summary())

# ================= Training =================

model.fit(
	train_generator,
	epochs=10,
	verbose=2
)


# ================= Results =================

predictions = model.predict(x=X_test, batch_size=BATCH_SIZE, verbose=0)
rounded_predictions = np.argmax(predictions, axis=-1)
correct = rounded_predictions==y_test

print(f'Classification report: \n{classification_report(y_test, rounded_predictions)}')
