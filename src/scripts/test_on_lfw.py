import sys
import os
import cv2
import time
import csv
# 获取当前脚本的绝对路径（scripts/下）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级得到项目根目录（假设 scripts/ 在 src/ 下，src/ 在项目根下）
project_root = os.path.dirname(os.path.dirname(current_dir))
# 将项目根目录加入搜索路径
sys.path.append(project_root)

# 现在可以按包结构导入

from core.face_engine import FaceEngine

def run_batch_test(data_dir, output_csv):
    # 1. 初始化引擎
    try:
        engine = FaceEngine()
        print("✅ 引擎加载成功，开始批量测试...")
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}")
        return

    # 2. 准备 CSV 报告文件
    header = ['文件夹', '文件名', '检测到人数', '耗时(秒)', '图片尺寸']
    results_data = []

    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 3. 遍历数据集目录
    # 建议数据集结构：data/场景名/图片
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                img_path = os.path.join(root, file)
                category = os.path.basename(root)
                
                # 读取图片
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                h, w = frame.shape[:2]
                
                # --- 核心测试环节 ---
                start_time = time.time()
                # 使用 is_static_image=True 确保调用最强的检测逻辑
                boxes, count = engine.detect_logic(frame, is_static_image=True)
                end_time = time.time()
                
                duration = round(end_time - start_time, 3)
                # ------------------

                results_data.append([category, file, count, duration, f"{w}x{h}"])
                print(f"已处理: [{category}] {file} | 找到: {count} | 耗时: {duration}s")

    # 4. 写入 CSV 结果
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results_data)

    print(f"\n✨ 测试完成！报告已保存至: {output_csv}")

if __name__ == "__main__":
    # 指定你的 200 张图片存放的总目录
    DATA_SET_PATH = "dataset" 
    # 输出报告文件名
    REPORT_NAME = "test_report2.csv"
    
    if not os.path.exists(DATA_SET_PATH):
        print(f"❌ 错误：找不到目录 {DATA_SET_PATH}，请先创建并放入图片。")
    else:
        run_batch_test(DATA_SET_PATH, REPORT_NAME)
