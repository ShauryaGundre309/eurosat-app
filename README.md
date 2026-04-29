# EuroSAT Land Classification App

## Overview
This project is developed for KaggleHacX ’26.  
The objective of the project is to classify satellite images into different land-use categories using Deep Learning and Computer Vision techniques.

The model is trained on the EuroSAT dataset using a pretrained MobileNetV2 architecture with Transfer Learning.

The application allows users to upload a satellite image and predicts the corresponding land category.

---

# Dataset Used

## EuroSAT Dataset
The EuroSAT dataset contains satellite images belonging to 10 different land-use and land-cover classes.

### Classes
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

---

# Model Used

## MobileNetV2
A pretrained MobileNetV2 model was used for transfer learning.

### Why MobileNetV2?
- Lightweight and fast
- Good accuracy
- Suitable for image classification tasks
- Efficient for deployment

---

# Training Details

## Image Size
224 x 224

## Training Strategy
- Transfer Learning
- Data Augmentation
- Train-validation split

## Loss Function
Categorical Crossentropy

## Optimizer
Adam

## Evaluation Metric
Accuracy

---

# Final Accuracy

Validation Accuracy Achieved:
88%+

---

# Features

- Upload satellite image
- Predict land category
- Display prediction confidence
- Lightweight deep learning model
- User-friendly interface

---

# Project Structure

```text
eurosat-app/
│
├── app.py
├── notebook.ipynb
├── eurosat_weights.weights.h5
├── requirements.txt
├── runtime.txt
├── README.md
