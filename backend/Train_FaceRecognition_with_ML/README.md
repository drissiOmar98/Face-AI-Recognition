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

| Notebook | Description |
|----------|-------------|
| [01_FRM_data_preprocessing_crop_faces](01_FRM_data_preprocessing_crop_faces.ipynb) | Preprocess raw dataset by detecting and **cropping faces** using OpenCV Haar Cascade. |
| 02_FRM_data_preprocessing_EDA | Perform **Exploratory Data Analysis (EDA)** on cropped dataset: visualize samples, check balance, analyze quality. |


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
