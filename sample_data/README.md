# AI River Biology：YOLOv8 中文版

这是一个面向 **River Dynamics and Integrated River Management** 课程大作业的 AI River Biology 原型 App。系统使用 **Streamlit + Ultralytics YOLOv8** 构建，用于鱼类和底栖动物图像的目标检测、类群识别、个体数量统计和结果报告导出。

## 1. 重要说明

本项目是真正调用 YOLOv8 的网页推理框架，但**不内置训练好的鱼类/底栖动物专用权重**。

- 默认 `yolov8n.pt`：只用于测试网页上传、检测框、计数和报表流程。
- 真正识别鱼类/底栖动物：需要使用你们自己标注并训练得到的 `best.pt`。
- 推荐展示表述：这是一个“基于 YOLOv8 的河流生物识别 App 原型”，后续通过现场标注数据训练专用权重提高精度。

## 2. 功能

- 中文界面；
- 单张图片 YOLOv8 检测；
- 批量图片 YOLOv8 检测；
- 检测框可视化；
- 类群数量统计；
- 采样点、河流、日期记录；
- CSV 和 TXT 报告下载；
- 自定义 `best.pt` 上传；
- YOLOv8 数据集配置和训练说明。

## 3. 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Streamlit Cloud 部署

1. 把本文件夹全部上传到 GitHub 仓库根目录；
2. 打开 Streamlit Cloud；
3. 选择仓库；
4. Main file path 填：

```text
app.py
```

5. 点击 Deploy。

## 5. 使用自定义 YOLOv8 权重

训练完成后，一般会得到：

```text
runs/detect/train/weights/best.pt
```

有两种使用方式：

1. 在 App 左侧选择“上传自定义 best.pt”；
2. 把 `best.pt` 上传到 GitHub 仓库的 `models/best.pt`，然后在 App 中选择“使用仓库内 models/best.pt”。

## 6. 推荐类群

鱼类：

- Cyprinidae fish（鲤科鱼类）
- Salmonidae fish（鲑科鱼类）
- Siluridae fish（鲇科鱼类）
- Cobitidae fish（鳅科鱼类）
- Gobiidae fish（虾虎鱼科鱼类）
- Centrarchidae fish（太阳鱼科鱼类）
- Percidae fish（鲈形目/鲈科近缘淡水鱼类）

底栖动物：

- Ephemeroptera nymph（蜉蝣目稚虫）
- Plecoptera nymph（襀翅目稚虫）
- Trichoptera larva（毛翅目幼虫）
- Chironomidae larva（摇蚊幼虫）
- Oligochaeta（寡毛类）
- Gastropoda（腹足类/螺类）
- Bivalvia（双壳类）
- Limnoperna fortunei（淡水壳菜/金贻贝）

## 7. 训练命令

准备好 YOLO 格式标注数据后：

```bash
yolo detect train data=config/river_biology_yolo_data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8
```

或者：

```bash
python scripts/train_yolov8.py
```

## 8. 作业展示话术

This app is a YOLOv8-based AI River Biology prototype for fish and benthic macroinvertebrate recognition. It accepts biological images, detects individual organisms, estimates counts, records sampling information, and exports CSV or text reports for river ecological assessment. The current version provides the full deployment and inference framework. With labelled field images and a custom trained `best.pt` model, it can be upgraded into a practical river bioimage recognition tool.
