# Skin Cancer Classification using CNN

## Overview
This project uses a **Convolutional Neural Network (CNN)** to classify images of skin cancer. The model is trained on a dataset of skin images from the International Skin Imaging Collaboration (ISIC) to detect different types of skin cancer. The implementation is done in **Python** using **TensorFlow/Keras**.

---

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information

This project is a Convolutional Neural Network (CNN)-based classification model designed to detect different types of skin cancer from medical images. The model is trained using a dataset of labeled skin images and aims to provide an automated approach for early skin cancer (melanoma)  detection accurately.

### Key Highlights
Uses deep learning techniques to classify skin lesions.
Built with TensorFlow and Keras.
Dataset includes multiple skin cancer types, preprocessed for model training.
Evaluates performance using accuracy, loss, and confusion matrix.
Potential for real-world applications in dermatology and medical diagnosis.

---

## **Dataset Overview**
- The dataset contains images of different types of skin cancer.
- It is preprocessed before being used for training the CNN model.

## **Key Steps**

### 1. **Data Loading & Preprocessing**
   - Load the skin cancer dataset (image files and labels).
   - Perform image resizing and normalization for better model training.
   - Apply data augmentation (rotation, flipping, zooming) using Augmentor.
   - Split the dataset into training, validation, and test sets.
### 2. **Build Convolutional Neural Network (CNN) Model**
   - Define a CNN architecture using Keras & TensorFlow.
   - Model consists of:
   - Convolutional layers (feature extraction)
   - Pooling layers (dimensionality reduction)
   - Fully connected layers (classification)
   - Softmax activation (output layer for multi-class classification)
   - Compile the model using:
   - Loss function: categorical_crossentropy
   - Optimizer: Adam
   - Evaluation metric: accuracy
### 3. **Model Training & Evaluation**
   - Train the model on the preprocessed dataset.
   - Monitor training & validation loss/accuracy.
   - Use early stopping and model checkpointing to prevent overfitting.
   - Evaluate model performance on the test dataset using:
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
### 4. **Model Predictions & Visualization**
   - Use the trained model to predict skin cancer types on test images.
   - Visualize predictions using Matplotlib (e.g., sample images with predicted labels).
   - Analyze misclassified images to improve model performance.

## Model Architecture
The CNN model consists of:
- **Convolutional Layers**: Extract features from images.
- **Pooling Layers**: Reduce dimensionality while retaining important features.
- **Fully Connected Layers**: Make final predictions based on extracted features.
- **Softmax Activation**: Used for classification.

## Results
- The final trained model achieves a high accuracy in classifying skin lesions.
- Performance metrics indicate the effectiveness of CNN in detecting skin cancer types.

---
  
## Conclusions
This case study provided analysis of skin lesions.

---

## Technologies Used

The project was built using the following technologies and libraries:
- **Python**: For data manipulation, analysis, and modeling.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Scikit-learn**: Provides utilities for data preprocessing, model evaluation, and performance metrics.
- **Matplotlib & Seaborn**: For data visualization.
- **statsmodels & sklearn**: Statistical modeling and machine learning.
- **Google Colab**: For interactive code development and documentation. Also, provides GPU for computation.
- **TensorFlow**: Provides backend support for building and training the CNN model.
- **Keras**: High-level API for constructing deep learning models.
- **OS**: Helps with file management and dataset handling.
- **Augmentor**: For image data augmentation (used when dataset is small).

---

## Acknowledgements

- This project was inspired by Upgrad.
- This project was based on [Upgrad Learning](https://www.upgrad.com).

---

## Contact
Created by [@JuturuKalyanSagar] - feel free to contact me!