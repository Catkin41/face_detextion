import sys
import os
import cv2
import dlib
import traceback
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# ==========================================
# 算法逻辑：Dlib CNN
# ==========================================
class SimpleFaceEngine:
    def __init__(self):
        # 获取当前脚本所在目录
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_path, "mmod_human_face_detector.dat")
        
        print(f"🔎 正在检查模型路径: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"未找到模型文件！请确保 .dat 文件和此脚本在同一文件夹。\n当前路径: {base_path}")
            
        print("🚀 加载 Dlib CNN 模型中...")
        self.detector = dlib.cnn_face_detection_model_v1(self.model_path)
        print("✅ 模型加载成功")

    def detect(self, frame):
        if frame is None: return [], 0
        # 预处理：缩放以提升识别率和速度
        h, w = frame.shape[:2]
        scale = 600 / w
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 检测 (0 表示不进行额外上采样)
        rects = self.detector(rgb_frame, 0)
        
        results = []
        for r in rects:
            rect = r.rect
            # 还原坐标
            l, t, rb, b = int(rect.left() / scale), int(rect.top() / scale), \
                           int(rect.right() / scale), int(rect.bottom() / scale)
            results.append((l, t, rb, b))
        return results, len(results)

# ==========================================
# 界面逻辑：PyQt5
# ==========================================
class FaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸检测追踪系统 (单文件版)")
        self.setFixedSize(800, 700)
        
        try:
            self.engine = SimpleFaceEngine()
        except Exception as e:
            # 弹窗报错，防止静默闪退
            QMessageBox.critical(self, "初始化错误", str(e))
            print(traceback.format_exc())
            sys.exit(1)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "警告", "无法打开摄像头！")

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_ui(self):
        layout = QVBoxLayout()
        self.info_lbl = QLabel("状态: 待机 | 人脸数: 0")
        self.info_lbl.setStyleSheet("font-size: 18px; color: blue; padding: 5px;")
        
        self.video_lbl = QLabel("画面待开启")
        self.video_lbl.setAlignment(Qt.AlignCenter)
        self.video_lbl.setStyleSheet("background: black; border: 2px solid gray;")
        self.video_lbl.setFixedSize(760, 500)
        
        self.btn = QPushButton("开启实时追踪")
        self.btn.setFixedHeight(40)
        self.btn.clicked.connect(self.toggle)
        
        layout.addWidget(self.info_lbl)
        layout.addWidget(self.video_lbl)
        layout.addWidget(self.btn)
        
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn.setText("开启实时追踪")
        else:
            self.timer.start(30)
            self.btn.setText("停止追踪")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            boxes, count = self.engine.detect(frame)
            
            for (l, t, r, b) in boxes:
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            
            self.info_lbl.setText(f"状态: 运行中 | 人脸数: {count}")
            self.display(frame)

    def display(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888)
        self.video_lbl.setPixmap(QPixmap.fromImage(qimg).scaled(self.video_lbl.width(), self.video_lbl.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    # 解决部分电脑上插件路径丢失的问题
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "" 
    
    app = QApplication(sys.argv)
    # 强制启用高清缩放防止窗口渲染异常
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    try:
        ex = FaceApp()
        ex.show()
        print("✨ 应用已启动")
        sys.exit(app.exec_())
    except Exception:
        print("🚨 发生未捕获异常：")
        print(traceback.format_exc())
        input("按回车键退出...") # 保持窗口，防止闪退看不到报错
