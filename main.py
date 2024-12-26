import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Set dataset path - Update this path to point to the chest_xray directory
dataset_path = r'C:\Users\prase\Downloads\archive(2)\chest_xray'

# Check if dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# Verify the folder structure
print("Folders under dataset path:")
print(os.listdir(dataset_path))

# Define paths for train and test
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

# Debug path existence
print(f"Train Path Exists: {os.path.exists(train_dir)}")
print(f"Test Path Exists: {os.path.exists(test_dir)}")

# Stop execution if paths are incorrect
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Train directory not found at {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found at {test_dir}")

# Split the training data to create a validation dataset (80-20 split)
validation_split = 0.2
batch_size = 32
img_size = (180, 180)

train_dataset = image_dataset_from_directory(train_dir,
                                             batch_size=batch_size,
                                             image_size=img_size,
                                             validation_split=validation_split,
                                             subset="training",
                                             seed=123)

val_dataset = image_dataset_from_directory(train_dir,
                                           batch_size=batch_size,
                                           image_size=img_size,
                                           validation_split=validation_split,
                                           subset="validation",
                                           seed=123)

# Load test dataset
test_dataset = image_dataset_from_directory(test_dir,
                                            batch_size=batch_size,
                                            image_size=img_size)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Visualize some sample images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
plt.show()

model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 10
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test Accuracy: {test_acc:.2f}')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

def predict_image(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(180, 180))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0]
    if score > 0.5:
        print("Prediction: Pneumonia Detected")
    else:
        print("Prediction: Normal")


