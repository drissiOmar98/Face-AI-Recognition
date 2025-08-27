import numpy as np
import sklearn
import pickle
import cv2


# Load all models
haar = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml') # cascade classifier
model_svm =  pickle.load(open('./models/model_svm.pickle', mode='rb')) # machine learning model (SVM)
pca_models = pickle.load(open('./models/pca_dict.pickle', mode='rb')) # pca dictionary
model_pca = pca_models['pca'] # PCA model
mean_face_arr = pca_models['mean_face'] # Mean Face


def faceRecognitionPipeline(filename,path=True):
    # Step-01: Read image (BGR)
    if path:
        # step-01: read image
        img = cv2.imread(filename)  # BGR
    else:
        img = filename  # array
    # Step-02: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step-03: Detect faces using Haar Cascade
    faces = haar.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

    # Initialize list to store predictions
    predictions = []

    # Process each detected face
    for x, y, w, h in faces:

        # Crop the face region
        roi = gray[y:y + h, x:x + w]

        # Step-04: Normalize pixel values to 0-1
        roi = roi / 255.0

        # Step-05: Resize to 100x100
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_CUBIC)

        # Step-06: Flatten image (1x10000)
        roi_flatten = roi_resize.reshape(1, 10000)

        # Step-07: Subtract mean face
        roi_mean = roi_flatten - mean_face_arr

        # Step-08: PCA transformation (Eigenface)
        eigen_image = model_pca.transform(roi_mean)

        # Step-09: Inverse PCA for visualization (optional)
        eig_img = model_pca.inverse_transform(eigen_image)

        # Step-10: Predict gender using SVM
        result = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()

        # Step-11: Annotate image with prediction
        text = f"{result[0]} : {int(prob_score_max * 100)}%"
        color = (255, 255, 0) if result[0] == 'male' else (255, 0, 255)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)

        # Store prediction info
        output = {
            'roi': roi,
            'eig_img': eig_img,
            'prediction_name': result[0],
            'score': prob_score_max
        }
        predictions.append(output)

    return img, predictions