[![Dataset: Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blueviolet)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)
[![Model: MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-brightgreen)](https://arxiv.org/abs/1801.04381)
[![Framework: TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)](https://www.tensorflow.org/)
[![Accuracy: 97%](https://img.shields.io/badge/Accuracy-97%25-success)](#-results)
[![F1 Score: 0.97](https://img.shields.io/badge/Macro%20F1--Score-0.97-blue)](#-results)

# Plant Disease Classification using CNN & Transfer Learning (MobileNetV2)
This project focuses on classifying **plant diseases** using **Transfer Learning** with **MobileNetV2**. It leverages a high-quality, real-world image dataset and trains a robust CNN pipeline capable of accurately classifying **38 plant disease classes**.

---

## Dataset
**Kaggle Plant Disease Dataset (38 Classes)**  
[Download from Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)  

- Total training images: `70,295`  
- Total validation images: `17,572`  
- 38 total classes, including multiple diseases per crop and healthy samples.

> Due to size restrictions, the dataset is **not included** in this repository.  
> Please **download it manually** from the Kaggle link above and place the folders as shown below:
> 
> The dataset directory for the current setup is - (project root)/dataset/(kaggle dataset)

---
## Project Pipeline

### 1. **Image Preprocessing**
- Real-time data augmentation using `ImageDataGenerator`
- Techniques: rotation, zoom, and horizontal flipping
- Input resolution: `224x224`

### 2. **Model Architecture**
This project uses **Transfer Learning** with the pretrained **MobileNetV2** architecture. Here's the breakdown:

- **Base Model**: `MobileNetV2` (ImageNet weights, `include_top=False`)
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dropout(0.3)`
  - `Dense(128, ReLU)`
  - `Dense(38, Softmax)` – final classification layer

### 3. Transfer Learning Process
- Phase 1: Base model frozen — trained only top layers
- Phase 2: Last 30 layers unfrozen — fine-tuned entire model

---

## Training Pipeline

- **Data Augmentation**: `rotation_range=20`, `zoom_range=0.2`, `horizontal_flip=True`
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 20
- **Callbacks**:
  - `EarlyStopping` (patience=5)
  - `ReduceLROnPlateau` (factor=0.2, patience=2)
  - `ModelCheckpoint` (saves the best model based on validation loss)

---

## Results

| Metric                  | Score |
|-------------------------|-------|
| **Validation Accuracy** | 97.72% |
| **Macro F1-Score**      | 0.9771  |
| **Validation Loss**     | 0.0664 |
| **Model**               | Saved as `.keras` |

---

## Results
- Excellent generalization
- Very high per-class precision & recall
- Strong performance on both disease & healthy samples
- Clean confusion matrix

---

## Outputs
- outputs/saved_model/best_model.keras
- Confusion Matrix plot
- Accuracy & Loss Curves

---

## Sample Predictions

| True Label             | Predicted Label         | Confidence (%) |
|------------------------|-------------------------|----------------|
| Apple___Apple_scab     | Apple___Apple_scab      | 100.0          |
| Apple___Apple_scab     | Apple___Apple_scab      | 100.0          |
| Apple___Apple_scab     | Apple___Apple_scab      | 100.0          |

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---

