# app/main.py

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cv2
import numpy as np
import os

from ultralytics import YOLO

import torch
from torch.serialization import add_safe_globals

# Импорт необходимых классов и функций для безопасной загрузки YOLO модели
from torch.nn.modules.container import Sequential, ModuleList
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import (
    Conv, C2f, Bottleneck, SPPF, Concat, Detect, DFL
)

# Разрешаем PyTorch безопасно десериализовать компоненты модели
add_safe_globals([
    DetectionModel,
    Conv, C2f, Bottleneck, SPPF, Concat, Detect, DFL,
    Sequential, ModuleList,
    Conv2d, BatchNorm2d, SiLU,
    MaxPool2d, Upsample
])

# Загружаем модель YOLO
model = YOLO("yolov8n.pt")

# Настройка FastAPI и шаблонов
app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def process_image(image: UploadFile = File(...)):
    print(f"[INFO] Получен файл: {image.filename}")
    contents = await image.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print("[ERROR] Не удалось декодировать изображение!")
        return {"error": "Некорректное изображение"}

    results = model(img)
    boxes = results[0].boxes
    if boxes is None:
        print("[INFO] Объекты не найдены.")
    else:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # координаты
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    result_path = "app/static/result.jpg"
    cv2.imwrite(result_path, img)
    print(f"[INFO] Сохранён результат в {result_path}")

    return {
        "detected_objects": len(boxes) if boxes else 0,
        "image_url": "/static/result.jpg"
    }
