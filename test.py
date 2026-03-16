import sys
import os
import cv2
import dlib
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

# ==========================================
# 核心算法模块：Dlib CNN (本地版)
# ==========================================
class FaceEngine:
    def __init__(self):
        import os
        print("当前工作目录：", os.getcwd())
        print("脚本所在目录：", os.path.dirname(os.path.abspath(__file__)))

        # 指定手动添加的模型文件名
        self.model_path = "mmod_human_face_detector.dat"
        
        if not os.path.exists(self.model_path):
            self.detector = None
            print(f"❌ 错误：在当前目录下未找到 {self.model_path}")
        else:
            print("🚀 正在加载 Dlib CNN 模型...")
            # 模块 3：加载深度学习人脸检测器
            self.detector = dlib.cnn_face_detection_model_v1(self.model_path)
            print("✅ 模型加载成功")

    def process_image(self, bgr_img):
        """
        包含预处理、检测、计数、可视化的核心流水线
        """
        if self.detector is None or bgr_img is None:
            return bgr_img, 0

        # --- 模块 2：图像预处理 ---
        # CNN 对大图处理较慢，若宽度超过 1200 像素则进行等比例缩小以提升速度
        h, w = bgr_img.shape[:2]
        scale = 1.0
        if w > 1200:
            scale = 1200 / w
            processing_img = cv2.resize(bgr_img, (0, 0), fx=scale, fy=scale)
        else:
            processing_img = bgr_img.copy()

        # Dlib 要求输入为 RGB 格式
        rgb_img = cv2.cvtColor(processing_img, cv2.COLOR_BGR2RGB)

        # --- 模块 3 & 4：人脸检测与优化 ---
        # 第二个参数 0 表示不对图像进行上采样，针对“大头人脸”效果最好且速度快
        # 若需检测极小的人脸，可改为 1
        rects = self.detector(rgb_img, 0)
        
        face_count = len(rects)
        
        # --- 模块 5：结果可视化 ---
        # 在原图上绘制结果，需将坐标映射回原始尺寸
        output_img = bgr_img.copy()
        for i, d in enumerate(rects):
            # 提取检测框坐标
            rect = d.rect
            l, t, r, b = int(rect.left() / scale), int(rect.top() / scale), \
                         int(rect.right() / scale), int(rect.bottom() / scale)
            
            # 针对“局部误判”和“遮挡”：Dlib CNN 会返回置信度
            # 置信度通常在 d.confidence 中，这里我们绘制绿色框
            cv2.rectangle(output_img, (l, t), (r, b), (0, 255, 0), 3)
            
            # 绘制序号
            cv2.putText(output_img, f"#{i+1}", (l, t - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return output_img, face_count

# ==========================================
# 界面模块：PyQt5 交互
# ==========================================
class FaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能人脸检测系统 - CNN 本地版")
        self.setGeometry(100, 100, 1000, 800)
        
        self.engine = FaceEngine()
        if self.engine.detector is None:
            QMessageBox.critical(self, "错误", "未找到模型文件 mmod_human_face_detector.dat，请检查目录！")
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        # 头部统计栏 (解决需求：显示人脸数量)
        self.info_label = QLabel("待机状态：请载入图像")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            background-color: #34495e; 
            color: #ecf0f1; 
            font-size: 20px; 
            font-weight: bold; 
            padding: 10px; 
            border-radius: 5px;
        """)
        main_layout.addWidget(self.info_label)

        # 图像显示区
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2c3e50; border: 2px solid #34495e;")
        main_layout.addWidget(self.image_label)

        # 底部按钮
        self.btn_load = QPushButton("读取本地图片进行检测")
        self.btn_load.setMinimumHeight(50)
        self.btn_load.setStyleSheet("font-size: 16px; background-color: #27ae60; color: white;")
        self.btn_load.clicked.connect(self.on_load_click)
        main_layout.addWidget(self.btn_load)

        central_widget.setLayout(main_layout)

    def on_load_click(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择人脸图片", "", "Images (*.jpg *.png *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is None:
                return

            # 执行核心检测
            processed_img, count = self.engine.process_image(img)
            
            # 更新计数显示
            self.info_label.setText(f"检测完成！查找到的人脸总数: {count}")
            
            # 刷新界面图像
            self.display_img(processed_img)

    def display_img(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(self.image_label.width(), 
                                                 self.image_label.height(), 
                                                 Qt.KeepAspectRatio, 
                                                 Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceApp()
    window.show()
    sys.exit(app.exec_())
