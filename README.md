# Brain Tumor Classification Using ResNet50, Vision Transformer and Hybrid ResNet-ViT

## Overview

This project presents a comparative study of three deep learning architectures for brain tumor classification from MRI images:

* ResNet50
* Vision Transformer (ViT)
* Hybrid ResNet-ViT

The objective is to classify MRI scans into four categories:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

The Hybrid ResNet-ViT model combines CNN-based local feature extraction with Transformer-based global feature learning, resulting in superior classification performance.

---

## Dataset

Dataset used:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Download the dataset and place it in the appropriate project directory before training or testing the models.

---

## Models and Results

| Model                    | Accuracy |
| ------------------------ | -------- |
| ResNet50                 | 70.87%   |
| Vision Transformer (ViT) | 77.36%   |
| Hybrid ResNet-ViT        | 97.57%   |

The Hybrid ResNet-ViT model achieved the highest accuracy and demonstrated the best overall performance for brain tumor classification.

---

## Technologies Used

* Python
* TensorFlow / Keras
* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Flask
* HTML
* CSS
* JavaScript

---

## Repository Contents

* Source code for ResNet50 model
* Source code for Vision Transformer model
* Source code for Hybrid ResNet-ViT model
* Flask web application
* Project report
* Documentation

---

## Deployment Note

This project was developed using Flask and deep learning models.

The GitHub Pages version serves as a frontend demonstration only. Since GitHub Pages does not support Python/Flask backend execution, model inference and dynamic template rendering are unavailable in the deployed version.

To run the complete application, execute the project locally using the provided source code.

---

## Trained Models

Trained model files are not included in this repository due to GitHub file size limitations.

The models can be recreated by running the training scripts provided in the repository.

---

## Project Report

The complete project report is included in this repository.
