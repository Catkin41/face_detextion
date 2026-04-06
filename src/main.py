import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from detection_manager import DetectionManager  # 检测逻辑管理器类


class VideoProcessorThread(QThread):
    frame_processed = pyqtSignal(object, list, int)  # 信号传递帧和检测结果

    def __init__(self, detection_manager, capture, parent=None):
        super().__init__(parent)
        self.detection_manager = detection_manager
        self.capture = capture
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.capture.read()  # 读取视频帧
            if ret:
                try:
                    frame = cv2.flip(frame, 1)
                    # 调用检测逻辑
                    boxes, count = self.detection_manager.process_frame(frame)
                    self.frame_processed.emit(frame, boxes, count)
                except Exception as e:
                    self.frame_processed.emit(None, [], 0)
                    QMessageBox.critical(None, "视频检测错误", str(e))

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能人脸识别系统")
        self.setFixedSize(1000, 800)

        self.init_ui()

        # 初始化逻辑管理器和视频线程
        self.detection_manager = DetectionManager()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法启动摄像头！请检查设备。")

        self.video_thread = VideoProcessorThread(self.detection_manager, self.cap)
        self.video_thread.frame_processed.connect(self.update_video_frame)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        # 状态标签
        self.status_lbl = QLabel("查找到的人脸数量: 0")
        self.status_lbl.setStyleSheet("font-size: 20px; color: #2ecc71; font-weight: bold; padding: 10px;")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_lbl)

        # 显示区域
        self.screen = QLabel("等待输入...")
        self.screen.setAlignment(Qt.AlignCenter)
        self.screen.setStyleSheet("background: #000; border: 2px solid #333;")
        self.screen.setMinimumSize(800, 550)
        layout.addWidget(self.screen)

        # 按钮区
        btn_layout = QHBoxLayout()
        self.btn_img = QPushButton("识别单张图片")
        self.btn_video = QPushButton("开启视频追踪")
        btn_layout.addWidget(self.btn_img)
        btn_layout.addWidget(self.btn_video)
        layout.addLayout(btn_layout)

        central.setLayout(layout)

        # 绑定按钮事件
        self.btn_img.clicked.connect(self.process_static_image)
        self.btn_video.clicked.connect(self.toggle_video)

    def process_static_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.png *.jpeg *.bmp)"
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "错误", "图片读取失败！请检查路径或图片格式。")
            return
        try:
            boxes, count = self.detection_manager.process_frame(frame, is_static_image=True)
            self.render_and_display(frame, boxes, count)
        except Exception as e:
            QMessageBox.critical(self, "检测错误", str(e))

    def toggle_video(self):
        if not self.video_thread.isRunning():
            self.video_thread.start()
            self.btn_video.setText("停止视频追踪")
        else:
            self.video_thread.stop()
            self.btn_video.setText("开启视频追踪")

    def update_video_frame(self, frame, boxes, count):
        if frame is not None:
            self.render_and_display(frame, boxes, count)

    def render_and_display(self, frame, boxes, count):
        for (l, t, r, b) in boxes:
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        self.status_lbl.setText(f"查找到的人脸数量: {count}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.screen.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.screen.width(), self.screen.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.video_thread.stop()
        self.cap.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())