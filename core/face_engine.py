import os
import cv2
import dlib
import numpy as np
# 导入新模块
from .config import Config
from .multi_detector import MultiAlgoDetector
from .result_fusion import ResultFusion

class FaceEngine:
    def __init__(self):
        # 1. 初始化多算法检测器（内部包含CNN、HOG等）
        try:
            self.detector_group = MultiAlgoDetector()
            self.fusion = ResultFusion()
        except Exception as e:
            raise RuntimeError(f"加载检测模块失败: {e}")
        
        # 2. 视频优化变量保持不变
        self.frame_counter = 0
        self.last_results = []
        self.last_count = 0
        self.skip_frames = 5 

    def _preprocess(self, frame):
        """保持原有的 CLAHE 增强逻辑，这对多人合影非常有效"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    def detect_logic(self, frame, is_static_image=False):
        if frame is None: return [], 0

        # 视频跳帧逻辑
        if not is_static_image:
            self.frame_counter += 1
            if self.frame_counter % self.skip_frames != 0 and self.last_results:
                return self.last_results, self.last_count

        h, w = frame.shape[:2]
        
        # 使用 Config 中的目标宽度进行动态缩放
        target_w = Config.TARGET_WIDTH if not is_static_image else 1600 # 静态图用更高分辨率
        scale = target_w / w if w > target_w else 1.0
        img_proc = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        rgb_img = self._preprocess(img_proc)

        # --- 修改点：调用多算法检测 ---
        # detect_all 会返回 [x1, y1, x2, y2, confidence] 的候选框列表
        proposals = self.detector_group.detect_all(rgb_img, scale)

        # --- 修改点：使用 NMS 结果融合 ---
        # 解决一张脸被不同算法重复标记的问题
        results = self.fusion.nms(proposals, Config.NMS_IOU_THRESHOLD)

        self.last_results = results
        self.last_count = len(results)
        return results, len(results)
