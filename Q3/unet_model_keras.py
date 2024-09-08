from keras import Sequential
from keras.src.layers import Conv2D, BatchNormalization, Dropout, Dense, Flatten, MaxPooling2D, preprocessing, Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.metrics.metrics_utils import confusion_matrix
from keras.src.utils import to_categorical
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import os
import keras
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
from keras import layers, models
from keras._tf_keras.keras.models import load_model

# K.tensorflow_backend._get_available_gpus()
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#
# keras.backend.set_session(sess)

# sess = tf.compat.v1.Session()
# K.set_session(sess)

# ============== Vars ==============

FULL_SIZE_IMG = 1  # set to 2 to use full size image
INPUT_SHAPE = (32, 32, 3)
num_classes = 4  # numb of classes in segmentation

# ============== Get Data ==============

def load_and_preprocess_image(image_path, target_size):
    """Load an image, resize it, and normalize it."""
    image = load_img(image_path, target_size=target_size, color_mode='grayscale')
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    return image


def load_data(image_folder, mask_folder, target_size):
    """Load and preprocess images and masks."""
    # sorted takes in an array.
    # the for loop creates an array with all the png names in a folder.
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

# Set the target size of the images

target_size = (128*FULL_SIZE_IMG, 128*FULL_SIZE_IMG)

# training data
train_images, train_masks = load_data('keras_png_slices_data/train/keras_png_slices_train', 'keras_png_slices_data/train/keras_png_slices_seg_train', target_size)

# testing data
test_images, test_masks = load_data('keras_png_slices_data/test/keras_png_slices_test', 'keras_png_slices_data/test/keras_png_slices_seg_test', target_size)

# For binary masks
train_masks = train_masks.astype(np.float32)
test_masks = test_masks.astype(np.float32)

# Convert masks to categorical one-hot encodings
train_masks = to_categorical(train_masks, num_classes=num_classes)
test_masks = to_categorical(test_masks, num_classes=num_classes)


# ============== Model ==============


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

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # TODO: make output onehot using sparse_categorical_crossentropy

    return model


model = unet_model()



# ============== Train or Load Model ==============

if os.path.isfile('models/unet_model_keras_128_res.h5') is False:
    print('Training Model')
    # Train
    history = model.fit(
        train_images,
        train_masks,
        validation_split=0.1,
        epochs=1,
        batch_size=16
    )

    model.save('models/unet_model_keras_128_res.h5')

else:
    print("Loading Model")
    model = load_model('models/unet_model_keras_128_res.h5')


# ============== Test Model ==============


# loss, accuracy = model.evaluate(test_images, test_masks)

predictions = model.predict(test_images)

# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")


fig, axes = plt.subplots(5,5, figsize = (10,10))
axes = axes.ravel()
for i in np.arange(0, 5*5):
    idx = np.random.randint(0, len(predictions))
    img = predictions[idx,1:]
    print(f"Image {i} - min: {img.min()}, max: {img.max()}")
    img_rescaled = minmax_scale(img.ravel(), feature_range=(0, 100)).reshape(img.shape)  # Make images more visable
    axes[i].imshow(img_rescaled, cmap='Blues', vmin=0, vmax=150)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()