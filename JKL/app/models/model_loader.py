import torch
from ultralytics import YOLO

def load_yolo_model(model_path="C:\\Users\\user\\Documents\\GitHub\\test\\JKL\\app\\models\\best_3000_xl.pt"):  
    """ YOLO 모델 로드 """
    try:
        model = YOLO(model_path)
        print(f"✅ YOLO 모델 로드 완료: {model_path}")
        return model
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        return None