# 直接部署步骤

## 第一步：上传到 GitHub

进入你的 GitHub 仓库 `ai-river-biology`，点击：

```text
Add file → Upload files
```

把本文件夹里的所有文件拖进去，然后点击：

```text
Commit changes
```

注意：`app.py` 必须在仓库最外层。

## 第二步：更新 Streamlit App

如果你已经部署过 Streamlit，上传 GitHub 后一般会自动重启。等 1–3 分钟，然后刷新原来的 App 链接。

如果没有自动更新，进入 Streamlit Cloud 后点击：

```text
Manage app → Reboot app
```

## 第三步：Main file path

Streamlit 部署时，主文件路径必须是：

```text
app.py
```

## 第四步：测试

1. 打开 App 首页；
2. 左侧进入“单张图片识别”；
3. 上传一张图片；
4. 点击“开始 YOLOv8 识别”；
5. 查看检测框、数量汇总和 CSV 下载。

## 重要提醒

默认 YOLOv8n 模型不是鱼类/底栖动物专用模型。要获得真正识别效果，请上传你们自己训练的 `best.pt`。
