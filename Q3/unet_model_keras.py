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
from sklearn.metrics import classification_report
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

FULL_SIZE_IMG = 2  # set to 2 to use full size image
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


# ============== Support Functions ==============

def recall_m(y_true, y_pred):
    """The number of TP results divided by everything that should have been identified as positive"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    """This is the number of true results divided by the total num of positive results"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    """This is the Dice or DSC or F1 Metric"""
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_coef(groundtruth_mask, pred_mask):
    # Adapted from: https://medium.com/@nghihuynh_37300/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f
    intersect = K.sum(pred_mask*groundtruth_mask)
    total_sum = K.sum(pred_mask) + K.sum(groundtruth_mask)
    dice = K.mean(2*intersect/total_sum)
    return dice #round up to 3 decimal places


# ============== Model ==============

def unet_model(input_size=(128*FULL_SIZE_IMG, 128*FULL_SIZE_IMG, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    en_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    en_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(en_conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(en_conv1)

    en_conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    en_conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(en_conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(en_conv2)

    # Bridge
    br_conv1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    br_conv1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(br_conv1)

    # Decoder
    up1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(br_conv1)
    concat1 = layers.concatenate([up1, en_conv2], axis=3)
    de_conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)
    de_conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(de_conv1)

    up1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(de_conv1)
    concat1 = layers.concatenate([up1, en_conv1], axis=3)
    de_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    de_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(de_conv1)

    outputs = layers.Conv2D(4, (1, 1), activation='sigmoid')(de_conv1)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', dice_coef])

    return model


model = unet_model()


# ============== Train / Load Model ==============

file_name = 'models/unet_model_keras_128_res_dice_double_depth.h5'

run_if_file_exists = True

if os.path.isfile(file_name) is run_if_file_exists:
    print('Training Model')
    # Train
    history = model.fit(
        train_images,
        train_masks,
        validation_split=0.1,
        epochs=1,
        batch_size=16
    )

    model.save(file_name)

else:
    print("Loading Model")
    try:
        model = load_model(file_name)
    except Exception as e:
        print(f"Couldn't find model to load. Full error:\n{e}" )


# ============== Test Model ==============


loss, accuracy, dsc = model.evaluate(test_images, test_masks)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

print(f"F1/DSC: {dsc}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")


# Predictions

predictions = model.predict(test_images)
y_pred_bool = np.argmax(predictions, axis=1)

print(f"y_pred_bool: {y_pred_bool}\n\nLabels: {test_masks}")

print(f"DSC: {dice_coef(test_masks, y_pred_bool)}")

# print(classification_report(test_masks, y_pred_bool))
print(f"Predictions: {y_pred_bool}\n\n One line: {y_pred_bool[0][0]}")

# Plot Predictions
fig, axes = plt.subplots(5,5, figsize = (10,10))
axes = axes.ravel()
for i in np.arange(0, 5*5):
    idx = np.random.randint(0, len(predictions))
    img = predictions[idx,1:]
    img_rescaled = minmax_scale(img.ravel(), feature_range=(0, 100)).reshape(img.shape)  # Make images more visable
    axes[i].imshow(img_rescaled, cmap='Blues', vmin=0, vmax=150)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()