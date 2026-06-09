# -*- coding: utf-8 -*-
"""
YOLOv8 训练脚本：AI River Biology
运行前请先准备好 YOLO 格式数据集，并修改 config/river_biology_yolo_data.yaml 中的 path。

命令行运行：
python scripts/train_yolov8.py
"""

from ultralytics import YOLO


def main():
    # 从官方 YOLOv8n 预训练权重开始微调。课程作业建议先用 n/s 小模型，速度快、部署轻量。
    model = YOLO("yolov8n.pt")
    model.train(
        data="config/river_biology_yolo_data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        project="runs/ai_river_biology",
        name="yolov8n_river_biology",
        patience=30,
    )


if __name__ == "__main__":
    main()
