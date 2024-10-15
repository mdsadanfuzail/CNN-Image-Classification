
# Convolutional Neural Network (CNN) for Image Classification

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images as either cats or dogs. It uses image data preprocessing techniques to augment the dataset and builds a CNN model to train, validate, and make predictions on unseen data.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Prediction](#prediction)
- [Results](#results)

## Introduction

This project demonstrates how to build a CNN for binary image classification using the TensorFlow and Keras libraries. The model is trained to classify images as either **cat** or **dog**, and the dataset is preprocessed using `ImageDataGenerator` for real-time data augmentation.

## Prerequisites

Make sure you have the following installed:
- Python 3.x
- TensorFlow 2.x
- Keras (comes with TensorFlow 2.x)
- Jupyter Notebook or Google Colab (optional for running the notebook)

You can install the required libraries using:
```bash
pip install tensorflow
```

## Project Structure

```
├── dataset/
│   ├── training_set/
│   │   ├── cats/
│   │   ├── dogs/
│   ├── test_set/
│   │   ├── cats/
│   │   ├── dogs/
├── cnn_image_classification.ipynb
├── README.md
```

- `dataset/`: Contains the training and test image datasets.
- `cnn_image_classification.ipynb`: Jupyter Notebook with the complete code for building, training, and evaluating the CNN model.
- `README.md`: Project documentation.

## Dataset

The dataset used is a subset of the [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data), consisting of images of cats and dogs in separate folders for training and testing.

- **Training set**: 10,000 images of cats and dogs (5000 each).
- **Test set**: 2,500 images (1250 each for cats and dogs).

You can organize the dataset as follows:
```
dataset/
├── training_set/
│   ├── cats/
│   ├── dogs/
├── test_set/
│   ├── cats/
│   ├── dogs/
```

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/cnn-image-classification.git
    ```

2. Navigate to the project directory:
    ```bash
    cd cnn-image-classification
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download and organize the dataset as shown in the [Dataset](#dataset) section.

## Model Architecture

The CNN architecture consists of the following layers:
1. **Convolution Layer**: 32 filters with 3x3 kernel and ReLU activation.
2. **Max Pooling**: 2x2 pooling layer to reduce the dimensionality.
3. **Second Convolution Layer**: Another Conv2D layer with 32 filters and MaxPooling.
4. **Flattening**: Convert the pooled feature maps into a 1D array.
5. **Dense Layer**: A fully connected layer with 128 units and ReLU activation.
6. **Output Layer**: A single neuron with sigmoid activation for binary classification.

## Training and Evaluation

1. Compile the model using the Adam optimizer and binary cross-entropy loss:
    ```python
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ```

2. Train the CNN on the training data and validate on the test data:
    ```python
    cnn.fit(x=training_set, validation_data=test_set, epochs=25)
    ```

## Prediction

To make a single prediction on a new image:
```python
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image / 255.0)
if result[0][0] > 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")
```

## Results

The model is trained for 25 epochs, achieving a training accuracy of around 90%. The validation accuracy is monitored to ensure the model generalizes well on the unseen test data.
