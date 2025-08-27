import os
import cv2
import matplotlib.image as matimg
from flask import Blueprint, request, jsonify

from app.face_recognition import faceRecognitionPipeline

UPLOAD_FOLDER = 'static/upload'
PREDICT_FOLDER = 'static/predict'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

api = Blueprint('api', __name__)

@api.route('/gender', methods=['POST'])
def gender_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    f = request.files['image']
    filename = f.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(file_path)

    pred_image, predictions = faceRecognitionPipeline(file_path)
    pred_filename = f'prediction_{filename}'
    pred_path = os.path.join(PREDICT_FOLDER, pred_filename)
    cv2.imwrite(pred_path, pred_image)

    # Prepare JSON report
    report = []
    for i, obj in enumerate(predictions):
        gray_name = f'roi_{i}_{filename}.jpg'
        eig_name = f'eigen_{i}_{filename}.jpg'
        matimg.imsave(os.path.join(PREDICT_FOLDER, gray_name), obj['roi'], cmap='gray')
        matimg.imsave(os.path.join(PREDICT_FOLDER, eig_name), obj['eig_img'].reshape(100,100), cmap='gray')

        report.append({
            "roi_image": f'/static/predict/{gray_name}',
            "eigen_image": f'/static/predict/{eig_name}',
            "gender": obj['prediction_name'],
            "score": round(obj['score']*100, 2)
        })

    return jsonify({
        "prediction_image": f'/static/predict/{pred_filename}',
        "faces": report
    })

@api.route('/test', methods=['GET'])
def test_home():
    return "API is working!"
