Pneumonia Detection Using CNN (TensorFlow/Keras)

This project uses Convolutional Neural Networks (CNN) to detect pneumonia from chest X-ray images. The dataset is the well-known chest X-ray dataset, which contains labeled images for training and testing.

Project Overview

The goal of this project is to develop a binary classification model that can differentiate between normal and pneumonia-affected chest X-ray images. TensorFlow and Keras are used to build, train, and evaluate the deep learning model.

Requirements

Python 3.7+

TensorFlow 2.0+

Matplotlib

Numpy

OS (built-in)

Installation

Install the required packages by running:

pip install tensorflow matplotlib numpy

Dataset

The dataset used in this project is organized into three folders:

train/ - Training images (with pneumonia and normal)

test/ - Test images (for final model evaluation)

val/ (Optional) - Validation images (if available)

Folder Structure

chest_xray/
  |- train/
  |- test/
  |- val/ (optional)

How to Run

Set Dataset PathUpdate the path to point to the chest X-ray dataset in the following line of the code:

 dataset_path = r'C:\path\to\chest_xray'

Check Dataset StructureThe script verifies the existence of train and test directories:

print(os.listdir(dataset_path))

Train-Validation SplitThe script automatically splits the training data into 80% training and 20% validation.

Model TrainingThe model is trained for 10 epochs. Adjust epochs as needed:

epochs = 10

EvaluationThe model is evaluated on the test dataset:

test_loss, test_acc = model.evaluate(test_dataset)

Prediction on New ImagesUse the following function to predict new chest X-ray images:

predict_image('path_to_image.jpg')

Model Architecture

Rescaling - Normalizes pixel values (0-1 range)

Conv2D - 3 convolutional layers with increasing filters (32, 64, 128)

MaxPooling2D - Pooling layers to reduce dimensionality

Flatten - Converts feature maps to a flat vector

Dense - Fully connected layers (128 neurons)

Dropout - Regularization to prevent overfitting

Output Layer - Sigmoid activation for binary classification

Visualization

Training and Validation Accuracy/LossTraining and validation accuracy/loss curves are plotted to monitor performance and detect overfitting.

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

Sample VisualizationVisualize 9 random images from the dataset:

plt.imshow(images[i].numpy().astype("uint8"))

Results

The model's accuracy on the test dataset is printed after evaluation.

Test accuracy is reported as:

print(f'Test Accuracy: {test_acc:.2f}')

Notes

Ensure the dataset is correctly labeled and placed in the specified directory structure.

Fine-tune the model by adjusting the number of epochs, batch size, or adding more convolutional layers.

Consider using data augmentation to improve generalization.

Acknowledgements

Dataset: Chest X-ray Images (Pneumonia)

Framework: TensorFlow/Keras.
