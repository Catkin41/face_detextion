# face_detextion

## 项目简介

本项目 `face_detextion` 是一个基于多种人脸检测方法的 Python 实现，旨在实现 **高精度、高召回率、多场景适用** 的人脸检测方案。支持经典算法与深度模型相融合，能够在标准、复杂、遮挡、多脸合影或大头场景中准确识别人脸，兼容各类图片数据，并具备良好的扩展性和易用性。

---

## 技术路线

### 1. **传统算法：OpenCV Haar/LBP 检测**
- 利用 OpenCV 的 Haar 和 LBP 特征分类器进行人脸检测，速度快，适用于标准人脸、低分辨率、实时需求场景。
- 提供极低计算资源消耗的检测方案。

### 2. **经典特征检测：dlib HOG 检测**
- dlib 的 HOG + SVM 检测方法对多尺度、复杂姿态和遮挡环境具备较强鲁棒性，召回率优于 Haar。
- 支持小脸和远景合影的人脸检测，提升整体召回率。

### 3. **深度学习模型：dlib CNN（mmod_human_face_detector）**
- 基于 dlib 的 CNN 人脸检测，支持复杂场景、遮挡、大头、侧脸、合影等，召回率和精度极高。
- 速度略慢，主要用于高精度需求和存储型检测场景。
- 模型文件需从官方渠道下载并存放于 `models/` 目录。

### 4. **多算法融合与非极大值抑制（NMS）**
- 各方法分别检测，结果融合，NMS 去除重叠框，保障召回率和去重。
- 自动根据图片分辨率和场景切换不同算法参数以优化速度和效果。

---

## 整体设计思路

### 1. **模块化结构**

- `face_engine.py`：核心人脸检测引擎，封装多种算法统一调用接口。
- 支持命令行批量处理、GUI集成、API调用。

### 2. **融合方案**

- 多种检测方法输出集成，保障不漏检，兼顾速度。
- 大图片自动缩放，高速检测；支持复杂遮挡和小脸场景。

### 3. **可扩展性**

- 可集成摄像头实时检测、Flask API等扩展。
- 支持不同硬件（CPU/GPU），可自动识别模型文件情况。

### 4. **易用性与兼容性**

- 纯 Python 实现，pip 安装依赖，无复杂环境要求。
- 支持 Windows、Linux、macOS等主流操作系统。

---

## 环境依赖

- Python >= 3.6
- opencv-python
- dlib
- numpy
- pillow

安装依赖：
```sh
pip install -r requirements.txt
```
如需深度模型支持，需下载 [mmod_human_face_detector.dat](http://dlib.net/files/mmod_human_face_detector.dat.bz2) 放入 `models/` 文件夹。

---

## 使用方法

### 命令行测试

```sh
python face_engine.py test_face.jpg
```

### 集成到其他项目

调用核心 `FaceEngine.detect_faces(image)`，返回所有人脸框坐标与数量。

---

## 文件结构示例

```
face_detextion/
│
├── face_engine.py      # 人脸检测引擎（多算法融合）
├── models/
│   └── mmod_human_face_detector.dat  # 深度人脸模型（如需）
├── requirements.txt
├── README.md
└── (其他辅助脚本或GUI等)
```

---

## 适用场景

- 多人合影、复杂背景、遮挡、侧脸、人脸大头、小脸等图片
- 安防、教育、照片管理、AI视觉、图像数据分析等
- 需要高召回率与速度兼顾的实际项目

---

## 参考文献与链接

- [dlib 官方](http://dlib.net/)
- [OpenCV 官方](https://opencv.org/)
- [LBP 算法文档](https://docs.opencv.org/3.4/dc/dc3/tutorial_py_face_detection.html)
- 非极大值抑制相关算法 [NMS](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c)

---

## 贡献与反馈

欢迎提交 issue 与 PR、提出建议与需求！
