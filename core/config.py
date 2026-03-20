import os
import cv2

class Config:
    # 路径管理
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    # 模型路径
    CNN_MODEL_PATH = os.path.join(MODEL_DIR, "mmod_human_face_detector.dat")
    HAAR_MODEL_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    # 算法开关
    USE_CNN = True
    USE_HOG = True

    # --- 关键参数微调 ---
    CONFIDENCE_THRESHOLD = 0.5   # 降低门限，捕捉 7 人合影中的边缘小脸
    NMS_IOU_THRESHOLD = 0.5      # 提高 IoU，允许 7 人合影中人脸重叠
    TARGET_WIDTH = 1200          # 预处理宽度，保证小脸像素足够
