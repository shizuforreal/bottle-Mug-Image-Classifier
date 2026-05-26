# Bottle vs Mug Image Classifier

## Overview
A binary image classifier that distinguishes between bottles and mugs 
using Haar feature extraction and a Support Vector Machine (SVM).

## Results
- **Accuracy: 72.50%**
- Trained on 200 images (100 bottles, 100 mugs)
- 80/20 train-test split

## Classification Report
| Class   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Bottles | 0.75      | 0.71   | 0.73     |
| Mugs    | 0.70      | 0.74   | 0.72     |

## Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

## How It Works
1. Load grayscale images, resize to 64x64
2. Extract 128 Haar-like features per image
3. Train a linear SVM classifier
4. Evaluate with classification report and confusion matrix

## Tech Stack
Python, OpenCV, scikit-learn, scikit-image, NumPy, Matplotlib, Seaborn

## Note on Dataset
Dataset: 100 bottle images + 100 mug images collected manually.
