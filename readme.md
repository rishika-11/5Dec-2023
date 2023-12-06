# FLOWER RECOGNITION

# Overview
The repository contains code for the flower recognition using data set.The data set contains 4242 images of flowers.The images are divided into five classes: daisy, tulip, rose, sunflower, dandelion. For each class, there are approximately 800 photos. The photos are not in high resolution, approximately 320×240 pixels.The project is to  recognize the images using deep learning.

# Prerequisites
- Python 3.11
- Tensorflow
- Keras
- Sklearn
- Matplotlib
- Numpy
- Cv2

# Model Architecture
It initializes a sequential convolutional neural network (CNN) using Keras, comprising a 2D convolutional layer with 64 filters, a filter size of (3,3), 'same' padding, and Rectified Linear Unit (ReLU) activation. This is followed by a flatten layer to transform the convolutional output into a one-dimensional vector, and a dense layer with 5 neurons and softmax activation, suitable for multi-class classification tasks. The model is designed to process input images with dimensions (128, 128, 3), where 128 represents the spatial dimensions, and 3 corresponds to the RGB color channels. This architecture is commonly employed for image classification, particularly in scenarios involving multiple classes.
The code compiles a neural network model using the Adam optimizer with a learning rate of 0.0001, categorical crossentropy as the loss function, and accuracy as the evaluation metric. It then trains the model for 7 epochs using data augmentation applied by datagen.flow on the training set (X_train, y_train) with a batch size of 32, while validating on the test set (X_test, y_test). The training progress and validation results are stored in the history variable for further analysis or visualization.

# Network Topology 

# Input Layer 
- Implicitly defined by the shape of the input data (128x128 RGB images).

# Output Layer 
- Dense layer with 5 neurons (5 classes).
- Activation function: Softmax for multiclass classification.

#  Weights 
- Convolutional layer (Conv2D) and dense layer have trainable weights.

#  Activation functions
- ReLU (Rectified Linear Unit) for the Convolutional layer.
- Softmax for the Output layer.

#  Biases 
- Implicitly present in the Convolutional layer and Dense layer.
- Biases are updated along with weights during training.

# Code Explanation
The code is a Python script for building and training a convolutional neural network (CNN) using TensorFlow Keras for a flower recognition task. It begins by importing necessary libraries, including OpenCV for image processing and scikit-learn for data preprocessing. The script then reads images from a specified directory, converts them to RGB format, and resizes them to 128x128 pixels. The images are stored in the 'data' list, and their corresponding labels are stored in the 'label' list. The labels are encoded numerically using scikit-learn's LabelEncoder and then one-hot encoded. The data is normalized, and a train-test split is performed. The CNN model consists of a convolutional layer with 64 filters, a flattening layer, and a dense layer with softmax activation for classification. Data augmentation is implemented using the ImageDataGenerator. The model is compiled with the Adam optimizer, categorical crossentropy loss, and accuracy as the metric. The training process involves fitting the model to the augmented training data. Finally, a random selection of test images is plotted, showcasing true labels in green and predicted labels in red, providing a visual assessment of the model's performance.





