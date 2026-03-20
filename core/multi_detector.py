import dlib
import cv2
import os
import numpy as np
from .config import Config

class MultiAlgoDetector:
    def __init__(self):
        if not os.path.exists(Config.CNN_MODEL_PATH):
            raise FileNotFoundError(f"模型不存在: {Config.CNN_MODEL_PATH}")
            
        self.cnn_detector = dlib.cnn_face_detection_model_v1(Config.CNN_MODEL_PATH)
        self.hog_detector = dlib.get_frontal_face_detector()

    def detect_all(self, rgb_img, scale):
        all_proposals = []
        
        # 1. CNN 检测 (核心：upsample=1 解决小脸问题)
        if Config.USE_CNN:
            # 开启 1 级上采样
            rects = self.cnn_detector(rgb_img, 1)
            for r in rects:
                if r.confidence > Config.CONFIDENCE_THRESHOLD:
                    # 必须提取 .rect 属性，并除以 scale 还原坐标
                    all_proposals.append(self._parse_rect(r.rect, scale, r.confidence))

        # 2. HOG 检测 (作为补充)
        if Config.USE_HOG:
            h_rects, scores, _ = self.hog_detector.run(rgb_img, 0, 0)
            for i, r in enumerate(h_rects):
                if scores[i] > 0.0: # HOG 分数逻辑不同，大于0即可视为候选
                    all_proposals.append(self._parse_rect(r, scale, scores[i]))
                    
        return all_proposals

    def _parse_rect(self, rect, scale, score):
        """统一将 dlib rect 转换为 [x1, y1, x2, y2, score]"""
        return [
            int(rect.left() / scale),
            int(rect.top() / scale),
            int(rect.right() / scale),
            int(rect.bottom() / scale),
            float(score)
        ]
