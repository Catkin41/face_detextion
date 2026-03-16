import sys
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from imutils.object_detection import non_max_suppression

# --- 核心算法类：封装模块 2, 3, 4 ---
class FaceProcessor:
    def __init__(self):
        # 加载 Haar 级联分类器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def process_frame(self, frame):
        """执行预处理、检测、后处理全流程"""
        if frame is None: return None
        
        display_frame = frame.copy()
        
        # 2. 图像预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. 人脸检测
        rects = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # 4. 后处理与优化 (NMS)
        if len(rects) > 0:
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(boxes, probs=None, overlapThresh=0.3)

            # 5. 结果可视化绘制
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(display_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face", (xA, yA - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display_frame

# --- 视频采集线程：模块 1 ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# --- 界面交互类：模块 5 ---
class FaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能人脸检测系统 v1.0")
        self.processor = FaceProcessor()
        self.initUI()
        
    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout()

        # 图像显示区域
        self.image_label = QLabel("等待输入...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2c3e50; border: 2px solid #34495e;")
        self.image_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.image_label)

        # 控制按钮区域
        btn_layout = QHBoxLayout()
        self.btn_file = QPushButton("读取本地图片")
        self.btn_camera = QPushButton("开启实时监控")
        self.btn_stop = QPushButton("停止")
        
        for btn in [self.btn_file, self.btn_camera, self.btn_stop]:
            btn_layout.addWidget(btn)
        
        main_layout.addLayout(btn_layout)
        self.central_widget.setLayout(main_layout)

        # 信号槽连接
        self.btn_file.clicked.connect(self.open_image)
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        
        self.thread = None

    def cv_to_pixmap(self, cv_img):
        """将 OpenCV 图像转换为 QPixmap 以便在 Label 显示"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def open_image(self):
        self.stop_camera()
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image files (*.jpg *.gif *.png *.jpeg)')
        if fname:
            img = cv2.imread(fname)
            processed_img = self.processor.process_frame(img)
            self.image_label.setPixmap(self.cv_to_pixmap(processed_img))

    def update_frame(self, cv_img):
        processed_img = self.processor.process_frame(cv_img)
        self.image_label.setPixmap(self.cv_to_pixmap(processed_img))

    def start_camera(self):
        if self.thread is None:
            self.thread = VideoThread()
            self.thread.change_pixmap_signal.connect(self.update_frame)
            self.thread.start()
            self.btn_camera.setEnabled(False)

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.btn_camera.setEnabled(True)
            self.image_label.clear()
            self.image_label.setText("摄像头已关闭")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceApp()
    window.show()
    sys.exit(app.exec_())
