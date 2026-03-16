import sys
import os
# 获取当前脚本所在目录的父目录（即项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from core.face_engine import FaceEngine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能人脸识别系统 (图像+视频版)")
        self.setFixedSize(1000, 800)

        # 初始化引擎
        try:
            self.engine = FaceEngine()
        except Exception as e:
            QMessageBox.critical(self, "启动失败", f"初始化异常: {e}\n{traceback.format_exc()}")
            sys.exit(1)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_video_frame)

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        # 状态统计
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
        
        for b in [self.btn_img, self.btn_video]:
            b.setFixedHeight(50)
            btn_layout.addWidget(b)
        
        layout.addLayout(btn_layout)
        central.setLayout(layout)

        # 信号绑定
        self.btn_img.clicked.connect(self.process_static_image)
        self.btn_video.clicked.connect(self.toggle_video)

    def process_static_image(self):
        """模块 1 & 5：处理静态图像文件，支持连续点击检测"""
        # 1. 先停止视频计时器（防止摄像头干扰）
        self.stop_video()
        
        # 2. 弹出文件选择框
        # 注意：path 如果为空（点击了取消），则直接返回，不影响下一次操作
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.png *.jpeg *.bmp)"
        )
        
        if path:
            # 3. 读取图片
            frame = cv2.imread(path)
            if frame is None:
                QMessageBox.warning(self, "错误", "图片读取失败，请检查路径是否有中文或文件损坏")
                return
                
            # 4. 执行检测逻辑 (is_static_image=True 保证高精度)
            try:
                boxes, count = self.engine.detect_logic(frame, is_static_image=True)
                # 5. 渲染并显示结果
                self.render_and_display(frame, boxes, count)
                print(f"检测完成: {path}, 人脸数: {count}")
            except Exception as e:
                QMessageBox.critical(self, "识别错误", f"检测过程中发生异常: {e}")
        
        # 6. 关键：处理完后，确保按钮依然是可用状态（默认就是可用的）
        # 如果点击取消，逻辑会直接结束，下次点击依然会触发此函数


    def toggle_video(self):
        """模块 1 & 5：切换实时视频追踪"""
        if not self.timer.isActive():
            self.timer.start(30)
            self.btn_video.setText("停止视频追踪")
            self.btn_video.setStyleSheet("background: #e74c3c; color: white;")
        else:
            self.stop_video()

    def stop_video(self):
        self.timer.stop()
        self.btn_video.setText("开启视频追踪")
        self.btn_video.setStyleSheet("")

    def process_video_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            boxes, count = self.engine.detect_logic(frame)
            self.render_and_display(frame, boxes, count)

    def render_and_display(self, frame, boxes, count):
        """结果可视化"""
        for (l, t, r, b) in boxes:
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        
        self.status_lbl.setText(f"查找到的人脸数量: {count}")
        
        # 格式转换
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888)
        self.screen.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.screen.width(), self.screen.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())