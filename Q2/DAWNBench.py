import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src.layers import Conv2D, BatchNormalization, Dropout, Dense, Flatten, MaxPooling2D, preprocessing
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()



# VISULISE DATASET


print(f"X_train: {X_train.shape}") # 50000, 32, 32, 3
print(f"y_train: {y_train.shape}") # 50000, 1
print(f"X_test: {X_test.shape}")   # 10000, 32, 32, 3
print(f"y_test: {y_test.shape}")   # 10000, 1

lbls = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog',
                'Frog', 'Horse', 'Ship', 'Truck']

fig, axes = plt.subplots(5,5, figsize = (10,10))
axes = axes.ravel()
for i in np.arange(0, 5*5):
    idx = np.random.randint(0, len(X_train))
    axes[i].imshow(X_train[idx,1:])
    lbl_idx = int(y_train[idx])
    axes[i].set_title(lbls[lbl_idx], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)

# Check if balanced

classes, counts = np.unique(y_train, return_counts=True)
plt.barh(lbls, counts)
plt.title('Class distribution in training set')

plt.show()

# PreProcessing
norm = keras.layers.RandomCrop(200, 200, 4)

X_train = norm.call(X_train)
y_train = norm.call(y_train)

## data normalisation
# scale the image data between 1 and 0 to make it uniform and easier for the neral network to learn
X_train = X_train / 255.0
X_test = X_test / 255.0

## One hot encoding
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

## Training / validation
X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(X_train,
                                                y_train_cat,
                                                test_size=0.2,
                                                random_state=42)



## Data augmentation
batch_size = 64
data_generator = ImageDataGenerator(horizontal_flip=True)

train_generator = data_generator.flow(X_TRAIN, Y_TRAIN, batch_size).shuffle()  # Added shuffle




# Model

INPUT_SHAPE = (32, 32, 3)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


# Train Model
model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy']
              )

history = model.fit(train_generator,
                    epochs=1,
                    validation_data=(X_VAL, Y_VAL),
                    )

# Model eval

plt.figure(figsize=(12, 16))

plt.subplot(4, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='val_Loss')
plt.title('Loss')
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.legend()

## Confusion Matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)

con = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=lbls)

fig, ax = plt.subplots(figsize=(10, 10))
con = con.plot(xticks_rotation='vertical', ax=ax,cmap='summer')

plt.show()
