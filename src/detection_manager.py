import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from core.face_engine import FaceEngine

class DetectionManager:
    def __init__(self):
        self.engine = FaceEngine()

    def process_frame(self, frame, is_static_image=False):
        if frame is None:
            raise ValueError("输入的帧不能为空！")
        return self.engine.detect_logic(frame, is_static_image)