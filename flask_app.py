from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Загрузка модели

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Обработка изображения через YOLOv8
    results = model(img)
    output_img = results[0].plot()  # Визуализация bounding boxes

    # Сохранение результата
    cv2.imwrite('static/result.jpg', output_img)

    return jsonify(count=len(results[0].boxes))  # Возвращаем количество объектов

if __name__ == '__main__':
    app.run(debug=True)
