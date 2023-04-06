# DigitRecognizer
Kaggle Competition

Kaggle Competition - Digit Recognizer:

This repository contains the solution for the Kaggle competition "Digit Recognizer", where the goal is to correctly 
identify digits from a dataset of tens of thousands of handwritten images. The solution is based on a deep learning 
model built with TensorFlow.

Dataset:

The dataset consists of 42,000 labeled images for training and 28,000 images for testing. The images are grayscale, 
28x28 pixels, and centered to reduce preprocessing.


Solution:

The solution is based on a convolutional neural network (CNN) built with TensorFlow. 
The model architecture consists of multiple convolutional layers followed by max pooling and a 
few fully connected layers. The model was trained using the training set and validated using a validation set split 
from the training set. The final predictions were made on the test set and submitted to the competition.

Requirements:

Python 3.x
TensorFlow 2.x
NumPy
Pandas
Matplotlib
Jupyter Notebook

Usage:

To reproduce the solution, download the dataset from the Kaggle competition page and extract it to the data directory.
Then, run the DigitRecognizerDNN.ipynb Jupyter Notebook to train the model and generate predictions on the test set.

Results:

The model achieved an accuracy of 98.95% on the validation set and a score of 0.99246 on the Kaggle competition leaderboard.

Credits:

This solution was developed by Tanishq Dwivedi for the Kaggle competition "Digit Recognizer".

