import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os


def load_and_preprocess_image(image_path, target_size):
    """Load an image, resize it, and normalize it."""
    image = load_img(image_path, target_size=target_size, color_mode='grayscale')
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    return image


def load_data(image_folder, mask_folder, target_size):
    """Load and preprocess images and masks."""
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    mask_filenames = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

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

# Set the target size of the images (e.g., 128x128)
target_size = (128, 128)

# Load training data
train_images, train_masks = load_data('keras_png_slices_data/train/keras_png_slices_train', 'keras_png_slices_data/train/keras_png_slices_seg_train', target_size)

# Load testing data
test_images, test_masks = load_data('keras_png_slices_data/test/keras_png_slices_test', 'keras_png_slices_data/test/keras_png_slices_seg_test', target_size)

# For binary masks
train_masks = train_masks.astype(np.float32)
test_masks = test_masks.astype(np.float32)

num_classes = 2  # Number of classes in segmentation

# Convert masks to categorical
train_masks = to_categorical(train_masks, num_classes=num_classes)
test_masks = to_categorical(test_masks, num_classes=num_classes)

from keras import layers, models


def unet_model(input_size=(128, 128, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Bottleneck
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)

    # Decoder
    up1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2)
    concat1 = layers.concatenate([up1, conv1], axis=3)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv3)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


model = unet_model()

history = model.fit(
    train_images,
    train_masks,
    validation_split=0.1,
    epochs=10,
    batch_size=16
)

# Evaluate model
loss, accuracy = model.evaluate(test_images, test_masks)

# Make predictions
predictions = model.predict(test_images)

print(f"loss: {loss}")
print(f"acc: {accuracy}")
# print(f": {accuracy}")