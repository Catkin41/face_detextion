import os
import cv2
import dlib
import numpy as np

class FaceEngine:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "..", "models", "mmod_human_face_detector.dat")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"核心模型文件缺失: {self.model_path}")
            
        self.detector = dlib.cnn_face_detection_model_v1(self.model_path)
        
        # 性能与缓存
        self.frame_counter = 0
        self.last_results = []
        self.last_count = 0
        self.skip_frames = 5 

    def _preprocess(self, frame):
        """高质量预处理：增强人脸特征细节"""
        # 1. 灰度化并进行 CLAHE 增强细节（解决多人场景光照问题）
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 2. 转换 RGB
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    def detect_logic(self, frame, is_static_image=False):
        if frame is None: return [], 0

        # 视频跳帧逻辑
        if not is_static_image:
            self.frame_counter += 1
            if self.frame_counter % self.skip_frames != 0 and self.last_results:
                return self.last_results, self.last_count

        h, w = frame.shape[:2]
        
        # --- 策略修改：不再强制缩小到 800/1000 ---
        # 如果是 7 张人脸的图，分辨率通常较高。我们只在图像大于 1600px 时才做轻微缩小
        # 这样能保证小人脸拥有足够的像素（至少 40x40 像素）
        if w > 1600:
            scale = 1600 / w
            img_proc = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            scale = 1.0
            img_proc = frame.copy()

        rgb_img = self._preprocess(img_proc)

        # --- 策略修改：直接使用 1 级采样检测（牺牲一点时间，换取多人识别率） ---
        # 之前的“错误框”通常是因为图像缩太小后背景产生的伪影。
        # 保持较大比例 + upsample=1 是目前解决小脸识别最稳妥的办法。
        rects = self.detector(rgb_img, 1)

        results = []
        for r in rects:
            # 只有置信度大于 0.6 的才保留（解决误报错误框问题）
            if r.confidence > 0.6:
                rect = r.rect
                l = int(rect.left() / scale)
                t = int(rect.top() / scale)
                rb = int(rect.right() / scale)
                b = int(rect.bottom() / scale)
                
                # 边界裁剪
                l, t, rb, b = max(0, l), max(0, t), min(w, rb), min(h, b)
                results.append((l, t, rb, b))

        self.last_results = results
        self.last_count = len(results)
        return results, len(results)
