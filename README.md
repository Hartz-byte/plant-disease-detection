[![Dataset: Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blueviolet)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)
[![Model: MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-brightgreen)](https://arxiv.org/abs/1801.04381)
[![Framework: TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)](https://www.tensorflow.org/)
[![Accuracy: 96%](https://img.shields.io/badge/Accuracy-96%25-success)](#-results)
[![F1 Score: 0.96](https://img.shields.io/badge/Macro%20F1--Score-0.96-blue)](#-results)

# Plant Disease Classification using CNN & Transfer Learning (MobileNetV2)
This project focuses on building a **high-performance image classification model** that can accurately detect **38 types of plant diseases** (including healthy classes) using images of plant leaves. The model achieves **96% accuracy** and a **Macro F1-score of 0.96** using **transfer learning** with **MobileNetV2** — optimized for deployment on edge devices or mobile.

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

## Model Architecture

This project uses **Transfer Learning** with the pretrained **MobileNetV2** architecture. Here's the breakdown:

- **Base Model**: `MobileNetV2` (ImageNet weights, `include_top=False`)
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dropout(0.3)`
  - `Dense(128, ReLU)`
  - `Dense(38, Softmax)` – final classification layer

The base model's weights are **frozen** initially to train the custom top layers efficiently, and can be later unfrozen for fine-tuning if needed.

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
| **Validation Accuracy** | 96.0% |
| **Macro F1-Score**      | 0.96  |
| **Loss**                | ↓ 0.12 |
| **Model**               | Saved as `.keras` |

---

## Outputs

All trained artifacts and evaluation results are stored in the `outputs/` folder:

---

## Sample Predictions

| True Label             | Predicted Label         | Confidence (%) |
|------------------------|-------------------------|----------------|
| Apple___Apple_scab     | Apple___Apple_scab      | 100.0          |
| Apple___Apple_scab     | Apple___Apple_scab      | 99.98          |
| Apple___Apple_scab     | Apple___Apple_scab      | 96.04          |

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---

