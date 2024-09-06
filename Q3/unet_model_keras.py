from keras import Sequential
from keras.src.layers import Conv2D, BatchNormalization, Dropout, Dense, Flatten, MaxPooling2D, preprocessing, Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import os
import keras
import tensorflow as tf
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt

# K.tensorflow_backend._get_available_gpus()
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#
# keras.backend.set_session(sess)

sess = tf.compat.v1.Session()
K.set_session(sess)

FULL_SIZE_IMG = 1  # set to 2 to use full size image

def load_and_preprocess_image(image_path, target_size):
    """Load an image, resize it, and normalize it."""
    image = load_img(image_path, target_size=target_size, color_mode='grayscale')
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    return image


def load_data(image_folder, mask_folder, target_size):
    """Load and preprocess images and masks."""
    # sorted takes in an array.
    # the for loop creates an array with all the png names in a folder.
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    mask_filenames = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
    # print([f for f in os.listdir(mask_folder) if f.endswith('.png')])

    images = []
    masks = []

    for img_name, mask_name in zip(image_filenames, mask_filenames):
        img_path = os.path.join(image_folder, img_name)
        mask_path = os.path.join(mask_folder, mask_name)

        image = load_and_preprocess_image(img_path, target_size)
        mask = load_and_preprocess_image(mask_path, target_size)

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Set the target size of the images

target_size = (128*FULL_SIZE_IMG, 128*FULL_SIZE_IMG)

# training data
train_images, train_masks = load_data('keras_png_slices_data/train/keras_png_slices_train', 'keras_png_slices_data/train/keras_png_slices_seg_train', target_size)

# testing data
test_images, test_masks = load_data('keras_png_slices_data/test/keras_png_slices_test', 'keras_png_slices_data/test/keras_png_slices_seg_test', target_size)

# For binary masks
train_masks = train_masks.astype(np.float32)
test_masks = test_masks.astype(np.float32)

num_classes = 4  # numb of classes in segmentation

# Convert masks to categorical one-hot encodings
train_masks = to_categorical(train_masks, num_classes=num_classes)
test_masks = to_categorical(test_masks, num_classes=num_classes)



# TODO: see how lecturer does this in monday or **Wend** presentation & Improve >:)

INPUT_SHAPE = (32, 32, 3)

# def model():
#     inputs = Input(INPUT_SHAPE)
#     e1 = Conv2D(filters=16, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same')(inputs)
#     e1 = Conv2D(filters=16, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same')(e1)
#     p1 = MaxPooling2D(pool_size=(2, 2))(e1)


# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same'))
# model.add(BatchNormalization())
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
# model.summary()

from keras import layers, models


def unet_model(input_size=(128*FULL_SIZE_IMG, 128*FULL_SIZE_IMG, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Bridge
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)

    # Decoder
    up1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2)
    concat1 = layers.concatenate([up1, conv1], axis=3)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

    outputs = layers.Conv2D(4, (1, 1), activation='sigmoid')(conv3)
    #
    # flat = layers.Flatten()(conv3)
    # relu = layers.Dense(64, activation='relu')(flat)
    # outputs = layers.Dense(10, activation='softmax')(relu)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = unet_model()



# Train Model
# model.compile(loss='categorical_crossentropy',
#                optimizer='adam',
#                metrics=['accuracy']
#               )

history = model.fit(
    train_images,
    train_masks,
    validation_split=0.1,
    epochs=1,
    batch_size=16
)

# Evaluate model
loss, accuracy = model.evaluate(test_images, test_masks)

# Make predictions
predictions = model.predict(test_images)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

fig, axes = plt.subplots(5,5, figsize = (10,10))
axes = axes.ravel()
for i in np.arange(0, 5*5):
    idx = np.random.randint(0, len(predictions))
    axes[i].imshow(predictions[idx,1:])
    lbl_idx = int(predictions[idx])
    axes[i].set_title(predictions[lbl_idx], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()