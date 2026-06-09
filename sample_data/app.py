# -*- coding: utf-8 -*-
"""
AI River Biology - YOLOv8 中文版
基于 Streamlit + Ultralytics YOLOv8 的鱼类与底栖动物图像识别网页原型。

重要说明：
1. 本 App 的推理框架是真正调用 Ultralytics YOLO。
2. 要实现鱼类/底栖动物的准确识别，需要上传你们自己训练好的 best.pt。
3. 默认 yolov8n.pt 只用于验证网页、上传、检测框、计数和报表流程，不是鱼类/底栖动物专用模型。
"""

from __future__ import annotations

import io
import os
import tempfile
import textwrap
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


APP_DIR = Path(__file__).resolve().parent
CLASSES_PATH = APP_DIR / "classes.csv"
DATA_YAML_PATH = APP_DIR / "config" / "river_biology_yolo_data.yaml"
REPO_MODEL_PATH = APP_DIR / "models" / "best.pt"


st.set_page_config(
    page_title="AI River Biology - YOLOv8 中文版",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Basic data and helper functions
# -----------------------------

@st.cache_data
def load_class_table() -> pd.DataFrame:
    if CLASSES_PATH.exists():
        return pd.read_csv(CLASSES_PATH)
    return pd.DataFrame(
        columns=["id", "class_name", "中文名称", "类别", "生态解释"]
    )


CLASS_TABLE = load_class_table()
CLASS_NAME_TO_CN = dict(zip(CLASS_TABLE["class_name"], CLASS_TABLE["中文名称"])) if not CLASS_TABLE.empty else {}
CLASS_NAME_TO_NOTE = dict(zip(CLASS_TABLE["class_name"], CLASS_TABLE["生态解释"])) if not CLASS_TABLE.empty else {}


def normalize_class_name(name: str) -> str:
    """Make YOLO class names comparable to our class table."""
    return str(name).replace(" ", "_").strip()


def chinese_label(name: str) -> str:
    key = normalize_class_name(name)
    return CLASS_NAME_TO_CN.get(key, str(name))


def ecological_note(name: str) -> str:
    key = normalize_class_name(name)
    return CLASS_NAME_TO_NOTE.get(
        key,
        "该类群暂无内置生态解释。若使用 COCO 演示模型，该结果只能说明检测流程可运行，不能代表真实鱼类或底栖动物识别结论。",
    )


def pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def result_to_dataframe(result, image_name: str, site: str, river: str, sample_date: str) -> pd.DataFrame:
    rows = []
    names: Dict[int, str] = result.names if hasattr(result, "names") else {}
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(
            columns=[
                "image", "river", "site", "date", "class_id", "class_name", "中文名称",
                "confidence", "x1", "y1", "x2", "y2", "生态解释"
            ]
        )

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    for i, cls_id in enumerate(cls):
        raw_name = str(names.get(int(cls_id), str(cls_id)))
        rows.append(
            {
                "image": image_name,
                "river": river,
                "site": site,
                "date": sample_date,
                "class_id": int(cls_id),
                "class_name": raw_name,
                "中文名称": chinese_label(raw_name),
                "confidence": round(float(conf[i]), 4),
                "x1": round(float(xyxy[i][0]), 2),
                "y1": round(float(xyxy[i][1]), 2),
                "x2": round(float(xyxy[i][2]), 2),
                "y2": round(float(xyxy[i][3]), 2),
                "生态解释": ecological_note(raw_name),
            }
        )
    return pd.DataFrame(rows)


def count_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["中文名称", "class_name", "count", "mean_confidence"])
    return (
        df.groupby(["中文名称", "class_name"], as_index=False)
        .agg(count=("class_name", "size"), mean_confidence=("confidence", "mean"))
        .sort_values(["count", "mean_confidence"], ascending=False)
    )


def make_report_text(df: pd.DataFrame, site: str, river: str, sample_date: str, model_name: str) -> str:
    lines = [
        "AI River Biology - YOLOv8 识别报告",
        "=" * 42,
        f"河流/水体：{river or '未填写'}",
        f"采样点：{site or '未填写'}",
        f"采样日期：{sample_date or '未填写'}",
        f"模型：{model_name}",
        "",
    ]
    if df.empty:
        lines += ["未检测到目标。建议检查图片清晰度、目标大小、置信度阈值或模型权重。"]
    else:
        summary = count_summary(df)
        lines.append("识别汇总：")
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['中文名称']} ({row['class_name']}): "
                f"{int(row['count'])} 个，平均置信度 {row['mean_confidence']:.2f}"
            )
        lines.append("")
        lines.append("管理解释：")
        for name in summary["class_name"].tolist():
            lines.append(f"- {chinese_label(name)}：{ecological_note(name)}")
    return "\n".join(lines)


@st.cache_resource(show_spinner="正在加载 YOLOv8 模型……")
def load_yolo_model(model_path: str):
    if YOLO is None:
        raise RuntimeError(f"无法导入 ultralytics：{ULTRALYTICS_IMPORT_ERROR}")
    return YOLO(model_path)


def get_model_from_sidebar() -> Tuple[str, str, bool]:
    """Return model path, display name, whether it is default demo model."""
    st.sidebar.markdown("### 模型设置")
    model_source = st.sidebar.radio(
        "模型来源",
        ["默认 YOLOv8n 演示模型", "上传自定义 best.pt", "使用仓库内 models/best.pt"],
        index=0,
        help="鱼类/底栖动物准确识别必须使用你们自己训练好的 best.pt。",
    )

    is_demo = False
    if model_source == "默认 YOLOv8n 演示模型":
        # Ultralytics will download the official small COCO pretrained model if not cached.
        return "yolov8n.pt", "YOLOv8n 官方演示模型（非鱼类/底栖动物专用）", True

    if model_source == "使用仓库内 models/best.pt":
        if REPO_MODEL_PATH.exists():
            return str(REPO_MODEL_PATH), "仓库 models/best.pt 自定义模型", False
        st.sidebar.error("仓库内还没有 models/best.pt，请先上传训练好的权重文件。")
        return "yolov8n.pt", "YOLOv8n 官方演示模型（临时备用）", True

    uploaded_model = st.sidebar.file_uploader("上传训练好的 YOLOv8 权重文件（.pt）", type=["pt"])
    if uploaded_model is not None:
        temp_dir = Path(tempfile.gettempdir()) / "ai_river_biology_yolov8_models"
        temp_dir.mkdir(parents=True, exist_ok=True)
        model_path = temp_dir / uploaded_model.name
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        return str(model_path), f"上传模型：{uploaded_model.name}", False

    st.sidebar.info("尚未上传 best.pt，暂时使用 YOLOv8n 演示模型。")
    return "yolov8n.pt", "YOLOv8n 官方演示模型（临时备用）", True


def draw_header() -> None:
    st.markdown(
        """
        <div style="padding: 1.2rem 1.4rem; border-radius: 18px; background: linear-gradient(120deg, #E8F7F5 0%, #FFF8EA 100%); border: 1px solid #C7E4E0;">
            <h1 style="margin-bottom: 0.2rem; color:#063B3A;">🐟 AI River Biology：YOLOv8 鱼类与底栖动物识别 App</h1>
            <p style="font-size: 1.05rem; margin-bottom: 0; color:#244B4A;">
            上传河流生物图像，自动输出目标检测框、类群识别、数量统计、采样信息和生态解释报告。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


# -----------------------------
# Sidebar global settings
# -----------------------------

st.sidebar.title("AI River Biology")
page = st.sidebar.radio(
    "功能导航",
    ["首页", "单张图片识别", "批量图片识别", "模型训练说明", "识别类群库", "方法与限制"],
)

model_path, model_display_name, using_demo_model = get_model_from_sidebar()
conf_thres = st.sidebar.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)
iou_thres = st.sidebar.slider("NMS IoU 阈值", 0.10, 0.90, 0.45, 0.05)
imgsz = st.sidebar.select_slider("推理图像尺寸", options=[320, 416, 512, 640, 768, 960], value=640)

if using_demo_model:
    st.sidebar.warning(
        "当前是官方 COCO 演示模型，只能测试 YOLOv8 流程。要识别鱼类/底栖动物，请上传自定义 best.pt。"
    )
else:
    st.sidebar.success("当前使用自定义 YOLOv8 权重。")


# -----------------------------
# Pages
# -----------------------------

if page == "首页":
    draw_header()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("算法框架", "YOLOv8")
    c2.metric("输入", "河流生物图像")
    c3.metric("输出", "检测框 + 数量")
    c4.metric("报告", "CSV / TXT")

    st.markdown("### 这个网站能做什么")
    st.markdown(
        """
        本 App 面向 **AI River Biology** 课程大作业，用于把鱼类和底栖动物图像转换成可展示、可下载的识别结果。核心流程为：

        1. 上传现场采集或实验室拍摄的生物图片；
        2. 使用 YOLOv8 对图片中的目标进行检测；
        3. 输出每个目标的类别、置信度和边界框；
        4. 自动统计不同类群的个体数量；
        5. 结合采样点、日期和河流名称生成 CSV 和文字报告。
        """
    )

    st.info(
        "作业展示建议：先展示首页和流程，再用一张鱼类图片、一张底栖动物图片、一组批量图片分别演示检测、计数和结果下载。"
    )

    st.markdown("### 技术流程")
    st.markdown(
        """
        **输入图像 → YOLOv8 目标检测 → 边界框提取 → 类群识别 → 个体数量统计 → 生态解释与报告导出**
        """
    )

    st.markdown("### 当前模型状态")
    if using_demo_model:
        st.warning(
            "当前加载的是 YOLOv8n 官方演示模型。它不是针对淡水鱼类和底栖动物训练的模型，因此不能把输出当成真实生物识别结果。"
        )
    else:
        st.success(f"当前加载模型：{model_display_name}")

elif page == "单张图片识别":
    draw_header()
    st.markdown("### 单张图片识别")

    with st.expander("填写采样信息", expanded=True):
        col1, col2, col3 = st.columns(3)
        river = col1.text_input("河流 / 水体名称", value="")
        site = col2.text_input("采样点", value="")
        sample_date = col3.date_input("采样日期", value=date.today()).isoformat()

    uploaded_image = st.file_uploader(
        "上传鱼类或底栖动物图片",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="原始图片", use_container_width=True)

        run_button = st.button("开始 YOLOv8 识别", type="primary")
        if run_button:
            try:
                model = load_yolo_model(model_path)
                with st.spinner("正在进行 YOLOv8 推理……"):
                    results = model.predict(
                        source=pil_to_rgb_array(image),
                        conf=conf_thres,
                        iou=iou_thres,
                        imgsz=imgsz,
                        verbose=False,
                    )
                result = results[0]
                annotated_bgr = result.plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                df = result_to_dataframe(result, uploaded_image.name, site, river, sample_date)
                summary = count_summary(df)

                st.markdown("### 识别结果")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("检测目标数", int(len(df)))
                col_b.metric("识别类群数", int(df["class_name"].nunique()) if not df.empty else 0)
                col_c.metric("模型", model_display_name[:28] + ("..." if len(model_display_name) > 28 else ""))

                st.image(annotated_rgb, caption="YOLOv8 检测结果", use_container_width=True)

                if df.empty:
                    st.warning("未检测到目标。可以降低置信度阈值，或检查图片清晰度、目标大小和模型权重。")
                else:
                    st.markdown("#### 数量汇总")
                    st.dataframe(summary, use_container_width=True)
                    st.markdown("#### 逐目标检测结果")
                    st.dataframe(df, use_container_width=True)

                    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "下载 CSV 结果表",
                        data=csv_bytes,
                        file_name="ai_river_biology_yolov8_result.csv",
                        mime="text/csv",
                    )
                    report = make_report_text(df, site, river, sample_date, model_display_name)
                    st.download_button(
                        "下载 TXT 识别报告",
                        data=report.encode("utf-8"),
                        file_name="ai_river_biology_yolov8_report.txt",
                        mime="text/plain",
                    )
            except Exception as exc:
                st.error("识别失败。请检查模型文件、依赖安装或图片格式。")
                st.exception(exc)
    else:
        st.info("请先上传一张图片。")

elif page == "批量图片识别":
    draw_header()
    st.markdown("### 批量图片识别")

    with st.expander("填写统一采样信息", expanded=True):
        col1, col2, col3 = st.columns(3)
        river = col1.text_input("河流 / 水体名称", value="")
        site = col2.text_input("采样点 / 样品批次", value="")
        sample_date = col3.date_input("采样日期", value=date.today()).isoformat()

    uploaded_images = st.file_uploader(
        "上传多张图片",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_images:
        st.write(f"已上传 {len(uploaded_images)} 张图片。")
        if st.button("开始批量识别", type="primary"):
            all_rows: List[pd.DataFrame] = []
            preview_cols = st.columns(2)
            try:
                model = load_yolo_model(model_path)
                progress = st.progress(0)
                for idx, file in enumerate(uploaded_images):
                    image = Image.open(file).convert("RGB")
                    results = model.predict(
                        source=pil_to_rgb_array(image),
                        conf=conf_thres,
                        iou=iou_thres,
                        imgsz=imgsz,
                        verbose=False,
                    )
                    result = results[0]
                    df_i = result_to_dataframe(result, file.name, site, river, sample_date)
                    all_rows.append(df_i)

                    if idx < 2:
                        annotated_bgr = result.plot()
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        preview_cols[idx % 2].image(annotated_rgb, caption=f"预览：{file.name}", use_container_width=True)
                    progress.progress((idx + 1) / len(uploaded_images))

                final_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
                st.markdown("### 批量识别汇总")
                st.dataframe(count_summary(final_df), use_container_width=True)
                st.markdown("### 批量逐目标结果")
                st.dataframe(final_df, use_container_width=True)

                csv_bytes = final_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "下载批量 CSV 结果",
                    data=csv_bytes,
                    file_name="ai_river_biology_yolov8_batch_results.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error("批量识别失败。")
                st.exception(exc)
    else:
        st.info("请上传两张或更多图片进行批量测试。")

elif page == "模型训练说明":
    draw_header()
    st.markdown("### YOLOv8 自定义模型训练说明")
    st.warning(
        "真正的鱼类和底栖动物识别，需要用标注好的图片训练自定义 YOLOv8 模型。只下载真实图片但不画检测框，不能训练目标检测模型。"
    )

    st.markdown("#### 1. 数据集目录结构")
    st.code(
        textwrap.dedent(
            """
            datasets/river_biology/
            ├── images/
            │   ├── train/
            │   ├── val/
            │   └── test/
            └── labels/
                ├── train/
                ├── val/
                └── test/
            """
        ).strip(),
        language="text",
    )

    st.markdown("#### 2. 标注格式")
    st.markdown(
        "每张图片需要一个同名 `.txt` 标注文件，每一行代表一个目标，格式为："
    )
    st.code("class_id x_center y_center width height", language="text")
    st.markdown("其中坐标均为 0–1 之间的归一化数值。可以用 LabelImg、CVAT、Roboflow 等工具标注。")

    st.markdown("#### 3. data.yaml")
    if DATA_YAML_PATH.exists():
        yaml_text = DATA_YAML_PATH.read_text(encoding="utf-8")
        st.code(yaml_text, language="yaml")
        st.download_button(
            "下载 river_biology_yolo_data.yaml",
            data=yaml_text.encode("utf-8"),
            file_name="river_biology_yolo_data.yaml",
            mime="text/yaml",
        )

    st.markdown("#### 4. 训练命令")
    st.code(
        "yolo detect train data=config/river_biology_yolo_data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8",
        language="bash",
    )

    st.markdown("#### 5. 训练完成后怎么用")
    st.markdown(
        "训练结束后，通常会得到 `runs/detect/train/weights/best.pt`。把这个文件上传到本 App 左侧的“上传自定义 best.pt”，或者放到 GitHub 仓库的 `models/best.pt`，即可用于在线识别。"
    )

elif page == "识别类群库":
    draw_header()
    st.markdown("### 建议识别类群库")
    st.dataframe(CLASS_TABLE, use_container_width=True)
    st.markdown(
        "这些类群是课程展示中相对稳妥的识别层级：鱼类建议先做到科/类群，底栖动物建议先做到目/科/大类群。样本量足够后再细化到物种。"
    )

elif page == "方法与限制":
    draw_header()
    st.markdown("### 方法与限制")
    st.markdown(
        """
        #### 方法
        - 本 App 使用 Ultralytics YOLOv8 作为目标检测框架；
        - 输入为鱼类或底栖动物图片；
        - 输出为检测框、类别、置信度、数量统计和报告；
        - 支持上传自定义 `best.pt` 权重，因此后续可以替换为真正基于现场数据训练的模型。

        #### 限制
        - 如果没有自定义训练权重，默认 YOLOv8n 不能准确识别鱼类和底栖动物；
        - 从网上搜集真实图片只能作为素材来源，仍然需要人工标注检测框；
        - 复杂背景、水下反光、目标遮挡、显微镜尺度差异都会影响检测效果；
        - 课程展示中建议明确表述为“YOLOv8 原型系统”，不要夸大为成熟业务系统。

        #### 推荐展示表述
        本系统构建了一个基于 YOLOv8 的河流生物图像识别 App。它能够完成图片上传、目标检测、数量统计和结果导出。当前版本提供完整的推理和部署框架，后续可通过标注现场采集的鱼类和底栖动物图片训练专用权重，从而提升识别精度和实际应用价值。
        """
    )
