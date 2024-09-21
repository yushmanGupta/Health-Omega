# Health Oracle - AI-Based Medical Image Disease Prediction

**Health Oracle** is an AI-powered application designed to predict diseases from medical images using deep learning models. It can classify four types of diseases: **Brain Tumor**, **Kidney Disease**, **Lung Cancer**, and **Tuberculosis**. Users can upload medical images via a user-friendly web interface, and the system provides predictions based on trained deep learning models.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Model Details](#model-details)
- [Screenshots](#screenshots)
- 
## Introduction

This repository contains the code for a Flask-based backend and a simple HTML/JavaScript frontend that can predict diseases from uploaded images. The system utilizes trained Convolutional Neural Networks (CNNs) for disease detection. The models can predict:
- Brain Tumor (No Tumor or Tumor)
- Kidney Disease (Cyst, Kidney Stone, Normal, or Tumor)
- Lung Cancer (Adenocarcinoma, Large Cell Carcinoma, Normal, or Squamous Cell Carcinoma)
- Tuberculosis (No TB or TB)

## Features

- **User-Friendly Interface**: Upload images and view predictions with ease.
- **Four Disease Models**: Separate models trained for brain, kidney, lung, and TB.
- **Image Upload**: Upload medical images for real-time disease prediction.
- **Confidence Scores**: The system shows probabilities for each disease class.

## Technologies Used

- **Backend**: Flask, TensorFlow, Keras
- **Frontend**: HTML, CSS, JavaScript
- **Deep Learning Models**: CNN (Convolutional Neural Networks)
- **Deployment**: AWS EC2 (Amazon Web Services), Nginx, Gunicorn

## Project Structure
Data Collection:
•	Brain Tumor: MRI images collected from public datasets like Kaggle and other open medical repositories.
•	Kidney Disease: Medical images were sourced from datasets containing various kidney conditions like cysts and tumors.
•	Lung Cancer: CT scan images with labels for different lung cancer types were used, available from healthcare datasets.
•	Tuberculosis: Chest X-ray datasets from organizations like NIH or WHO provided images labeled as either TB-positive or TB-negative.
Data Preprocessing:
1.	Image Resizing: All input images were resized to 150x150 pixels to maintain uniformity across models and reduce computational complexity.
2.	Normalization: Pixel values were normalized by dividing by 255, bringing them into a 0-1 range, which improves the convergence of neural networks.
3.	Augmentation: Applied techniques such as rotation, zoom, horizontal flipping, and shearing to artificially expand the dataset and prevent overfitting.
- **Model Architecture:**
We utilized Convolutional Neural Networks (CNNs) for all models due to their strong performance in image recognition tasks. The architecture for each model was tweaked based on the dataset and complexity of the problem:
1.	Base Model: Each model starts with the same general structure:
o	Convolutional Layers: Extract key features like edges, textures, and patterns from the images. Multiple convolutional layers with ReLU activation were used.
o	Pooling Layers: Reduce the spatial dimensions of the image while retaining important features. MaxPooling was applied to down-sample the feature maps.
o	Dropout Layers: Added dropout after the convolutional layers to reduce the risk of overfitting by randomly deactivating neurons during training.
o	Dense Layers: The output from the convolutional layers was flattened and passed to one or more dense layers for classification.
2.	Fine-tuning the architecture:
o	Brain Tumor: A relatively simple binary classification problem (tumor/no tumor), so we used three convolutional layers, followed by two dense layers. Softmax activation was used in the final layer.
o	Kidney Disease: Since this was a multi-class classification (cyst, stone, tumor, normal), we added an additional convolutional block to enhance feature extraction.
o	Lung Cancer: Given the complex nature of distinguishing between various types of lung cancer, a deeper model with five convolutional layers was employed.
o	Tuberculosis: This is a binary classification problem, similar to brain tumor detection, but we fine-tuned it to work better with X-ray images using four convolutional layers.
- **Transfer Learning:**
For some models, transfer learning was applied to boost performance. Pre-trained models like VGG16 and ResNet50 were used as a backbone, and additional layers were trained on our dataset. This approach helped speed up the training process and improve accuracy, especially for the Lung Cancer and Tuberculosis models where large datasets are harder to obtain.
1.	VGG16: Pre-trained on ImageNet and fine-tuned for brain tumor and kidney disease detection.
2.	ResNet50: Used for lung cancer detection to capture deeper and more complex features of lung images.
## Screenshots
![image](https://github.com/user-attachments/assets/c8263c55-b373-4676-9faa-e5fd9d51d2c9)

![image](https://github.com/user-attachments/assets/4c6b7bfe-5e51-4818-8510-f95e912a58fa)
![image](https://github.com/user-attachments/assets/3dff82fd-83f7-46b8-afab-6b7d8692a8b7)
![image](https://github.com/user-attachments/assets/8ef17908-7b78-4e40-acf9-a8d595edd556)




