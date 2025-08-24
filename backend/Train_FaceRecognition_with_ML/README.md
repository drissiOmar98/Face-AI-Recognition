# 🧑‍💻 Face Recognition Model (FRM) with OpenCV & Machine Learning

This project aims to build a **Face Recognition Model** that can detect and classify gender (male/female) from images and videos.  
It uses **OpenCV** for computer vision tasks, traditional ML/DL models for classification, and **Jupyter Notebooks** to document each step of the pipeline.  

The project is organized into multiple notebooks, each focusing on a specific stage:  
- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Model training  
- Evaluation  
- Deployment  

---

## 📑 Table of Contents (Notebooks)

| Notebook                                                                          | Description                                                                                                                                                            |
|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [01_FRM_data_preprocessing_crop_faces](01_FRM_data_preprocessing_crop_faces.ipynb) | Preprocess raw dataset by detecting and **cropping faces** using OpenCV Haar Cascade.                                                                                  |
| [02_FRM_data_preprocessing_EDA](02_FRM_data_preprocessing_EDA.ipynb)       | Perform **Exploratory Data Analysis (EDA)** on cropped dataset: visualize samples, check balance, analyze quality.                                                     |
| [03_FRM_feature_extraction_eigenfaces](03_FRM_feature_extraction_eigen_face.ipynb) | Apply **PCA (Eigenfaces)** for dimensionality reduction: extract features, visualize eigenfaces, and save reduced dataset for ML models. |
| [04_FRM_Machine_Learning](04_FRM_Machine_Learning.ipynb)                           | Train and evaluate **Machine Learning models (SVM, Logistic Regression, Random Forest)** on PCA features for **gender classification**. |
| [05_FRM_Make_Pipeline](05_FRM_Make_Pipeline.ipynb)                                 | Build a complete **end-to-end pipeline**: face detection → preprocessing → PCA → SVM prediction. Prepares system for **Flask deployment**. |


---

## 📘 Notebook 01 — Data Preprocessing (Crop Faces)

In this notebook, we prepare the raw dataset for the **Face Recognition Model (FRM)** by detecting and cropping faces using **OpenCV** and its pre-trained **Haar Cascade Classifier**.

### 🔑 Key Steps
- Load dataset images (male/female)  
- Convert images from **BGR → RGB** for correct visualization  
- Apply Haar Cascade to **detect face regions**  
- **Crop detected faces** to retain only the relevant region (face)  
- Save cropped images into structured folders:  
  - `./data/crop_data/male/`  
  - `./data/crop_data/female/`

### 🎯 Outcome
- Raw images are now **cleaned and standardized** into cropped face images  
- This reduces noise and ensures the dataset is **ready for training and further analysis**
---

## 📘 Notebook 02 — Exploratory Data Analysis (EDA) on Cropped Dataset

In this notebook, we explore the **cropped face images** from Notebook 01 to better understand the dataset characteristics and prepare it for model training.

### 🔑 Objectives
1. **Distribution of Male and Female Images**  
   - Visualized counts using **Bar Chart**  
   - Visualized proportions using **Pie Chart**

2. **Distribution of Image Sizes (Width & Height)**  
   - Analyzed using **Histograms** and **Box Plots**  
   - Compared by **gender**

3. **Determine Target Image Dimensions for Resizing**  
   - Standardized all images to **100×100 pixels**  

4. **Filter Out Very Small Images**  
   - Removed images with height ≤ 60 pixels

5. **Image Transformation for ML Models**  
   - Converted to **grayscale**  
   - Resized to 100×100  
   - Flattened into **1D vectors (10,000 pixels)**  
   - Normalized pixel values from **[0–255] → [0–1]**

### 🎯 Outcome
- Dataset is **balanced** between male and female images  
- All images standardized to **100×100 grayscale vectors**  
- Pixel values normalized for stable training  
- Final dataset stored in `data/data_images_100_100.pickle` for future use  

### 📝 Key Insights
- Gender distribution is nearly balanced.  
- Female images tend to have higher resolution than male images.  
- Filtering and normalization ensured the dataset is **clean, consistent, and training-ready**.

---

## 📘 Notebook 03 — Feature Extraction with Eigenfaces (PCA)

In this notebook, we apply **Principal Component Analysis (PCA)**, also known as the **Eigenfaces method**, to reduce the dimensionality of our preprocessed face dataset.  
This allows us to capture the **most informative features** while reducing noise and computational cost.

### 🔑 Objectives
1. **Compute the Mean Face**  
   - Average face image across the dataset.  
   - Subtracted from each image for centering before PCA.

2. **Apply PCA (Eigenfaces)**  
   - Extracted principal components (Eigenfaces) that capture key facial variations.  
   - Reduced image vectors (10,000 features) to **50 principal components**, retaining ~80% of variance.

3. **Explained Variance Analysis**  
   - Plotted explained and cumulative variance ratios.  
   - Used **Elbow Method** to select optimal number of components.

4. **Visualization**  
   - Displayed the mean face, Eigenfaces, and reconstructed images.  
   - Compared **original vs reconstructed faces** using 50 components.

5. **Feature Extraction for Classification**  
   - Transformed dataset into **50-dimensional feature vectors**.  
   - Saved PCA features and labels to `./data/data_pca_50_target.npz`.  
   - Saved PCA model & mean face to `./models/pca_dict.pickle`.

### 🎯 Outcome
- Reduced **dimensionality** of dataset (10,000 → 50 features).  
- Extracted **informative Eigenface features** for each image.  
- Prepared dataset for **machine learning classification models**.  

---

## 📘 Notebook 04 — Machine Learning Models for Gender Classification  

In this notebook, we use the **PCA features (Eigenfaces)** extracted in Notebook 03 to train and evaluate different **Machine Learning classifiers** for gender prediction.  
The goal is to identify the best-performing model for building a reliable gender classification system.  

### 🔑 Objectives  
1. **Load PCA Features**  
   - Imported preprocessed dataset from `./data/data_pca_50_target.npz`.  
   - Used 50 principal components per image as input features.  

2. **Split Dataset**  
   - Divided into **training (80%)** and **testing (20%)** sets.  
   - Ensured balanced male/female representation.  

3. **Train Machine Learning Models**  
    - Apply classifiers such as **Support Vector Machine (SVM)**, Logistic Regression, Random Forest, etc.  

4. **Model Evaluation**  
   - Measured performance using:  
     - **Accuracy**.  
     - **Precision & Recall**.  
     - **Confusion Matrix & Classification Report**.  

5. **Save Best Model**  
   - Stored trained models in `./models/` for later inference.  
   - Prepared for integration with the Flask web application.  

### 🎯 Outcome
- **Best Model:** SVM with hyperparameter tuning (via `GridSearchCV`)  
- **Accuracy:** ~78.8%  
- **Performance Insights:**
  - Female faces classified slightly better than male faces.  
  - Balanced performance overall with Macro F1 ≈ 0.78.  

---
# 📘 05_FRM_Make_Pipeline

## 🔎 Introduction
Up until now, we have:
- **Cropped faces** from raw images.  
- Conducted **EDA** to assess dataset quality.  
- Extracted **Eigenfaces (PCA features)**.  
- Trained an **SVM model** for gender classification.  

This notebook integrates all of these steps into a **single end-to-end pipeline**.  
The goal is to ensure that any new input image can pass through the same **preprocessing → feature extraction → classification** workflow seamlessly.  

---

## 🎯 Objectives
1. **Face Detection & Preprocessing**  
   - Detect faces with OpenCV Haar Cascade.  
   - Convert to grayscale, resize to uniform shape, normalize, and flatten.  

2. **Feature Extraction (PCA - Eigenfaces)**  
   - Apply the trained PCA model to reduce dimensionality.  
   - Reconstruct eigenfaces for visualization.  

3. **Prediction with Trained Classifier**  
   - Use the best-performing **SVM model** (loaded from disk) to classify gender.  
   - Display results with labels (Male/Female).  

4. **Pipeline Integration**  
   - Combine preprocessing, PCA, and SVM into a unified **scikit-learn pipeline**.  
   - Save the pipeline for direct use in deployment.  

---

## 📊 Results
- Created a robust **face recognition pipeline** that transforms raw input into predictions.  
- Example predictions correctly displayed **bounding boxes and gender labels**.  
- Eigenfaces visualizations confirm that PCA captures key facial features.  

---




