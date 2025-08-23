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


