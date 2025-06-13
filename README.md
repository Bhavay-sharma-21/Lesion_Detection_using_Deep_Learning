# 🦷 Periapical Lesion Detection Using Deep Learning

## 📌 Project Overview
This project focuses on the automated detection of **periapical lesions** from dental radiographs using modern **deep learning models**. It utilizes pre-trained CNN architectures and transfer learning to perform binary classification — detecting whether a lesion is present or not.

---

## 🧠 Models Implemented

- ✅ VGG16  
- ✅ InceptionV3  
- ✅ ResNet50  
- ✅ EfficientNetB0  
- ✅ DenseNet121  

These models were fine-tuned with custom classification layers to suit the task.

---

## 📁 Dataset

- Trained the models on REAL_TIME_DATASET
- **Classes**: `Lesion`, `No Lesion`
- **Input Shape**: 224x224
- **Preprocessing**:
  - Rescaling / Normalization
  - Data Augmentation: Rotation, Flipping, Zoom

---

## ⚙️ Workflow

1. Data loading using `ImageDataGenerator`
2. Model building with transfer learning (ImageNet weights)
3. Custom classification head addition
4. Model training with:
   - Loss: `binary_crossentropy`
   - Optimizer: `Adam`
   - Metrics: Accuracy, Precision, Recall
5. Evaluation using:
   - Confusion matrix
   - ROC-AUC
   - Classification report

---

## 📊 Results Summary

| Model          | Accuracy | 
|----------------|----------|
| VGG16          | 71%      |
| InceptionV3    | 95.83%      | 
| ResNet50       | 79%      | 
| EfficientNetB0 | 50%      | 
| DenseNet121    | 95.93%      | 



---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/periapical-lesion-detection
cd periapical-lesion-detection

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook VGG16.ipynb
