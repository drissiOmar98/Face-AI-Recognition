# 🧑‍💻 Face Recognition API (Flask Backend)  

This is the backend API for a Face Recognition project with gender detection.  
It provides REST endpoints for uploading images, processing them with a PCA + SVM-based gender recognition pipeline, and returning predictions along with annotated images.  

The backend is built with **Flask**, uses **CORS** to allow requests from an Angular frontend, and organizes logic in blueprints for scalability.

<p align="center">
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-F58025?style=for-the-badge&logo=matplotlib&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit-Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
</p>  

---

## ✨ Features

- 🖼 Upload and analyze images for gender detection.  
- 📊 Return annotated images and prediction reports in JSON format.  
- 🌐 CORS enabled for Angular frontend integration.  
- 🧪 Test endpoint to verify API connectivity.  

---


## 📂 Project Structure

```text
flaskApp/
├── app/
│   ├── routes.py               # 🛣 API routes blueprint (/gender, /test)
│   └──face_recognition.py     # 🧠 Face recognition pipeline (Haar cascades, PCA, SVM)
│   
├── models/
│   ├── haarcascade_frontalface_default.xml  # 🖼 Haar cascade for face detection
│   ├── model_svm.pickle                     # 🧩 Trained SVM model for gender classification
│   └── pca_dict.pickle                      # 🔹 PCA model and mean face array
├── static/
│   ├── upload/                 # 📤 Stores images uploaded by users
│   └── predict/                # 🔍 Stores prediction results (annotated images, ROI, eigenfaces)
├── main.py                      # 🚀 Main Flask entry point
├── requirements.txt            # 📦 Python dependencies
└── README.md                   # 📄 Project documentation
