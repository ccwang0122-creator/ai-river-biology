from __future__ import annotations

import io
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageDraw

# =========================
# 基础路径
# =========================
BASE_DIR = Path(__file__).parent
REF_DIR = BASE_DIR / "reference_images"
REF_ZIP = BASE_DIR / "reference_images.zip"
MODEL_PATH = BASE_DIR / "model" / "reference_features.npz"
META_PATH = BASE_DIR / "reference_metadata.csv"
SUMMARY_PATH = BASE_DIR / "label_summary.csv"
LATIN_PATH = BASE_DIR / "latin_names.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

st.set_page_config(
    page_title="淡水浮游植物识别 App",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# 样式
# =========================
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(135deg, #eef9f6 0%, #fff9ed 100%);
        border: 1px solid #d5ebe3;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0 0 .6rem 0; color:#0b3d4a;}
    .tag {
        display:inline-block; padding:.28rem .62rem; border-radius:999px;
        border:1px solid #b7dcec; background:#f3fbff; color:#075985;
        font-size:.86rem; margin:.15rem .25rem .15rem 0;
    }
    .note {
        padding: .9rem 1rem; border-radius: 12px; background:#fff8db;
        border:1px solid #f0dc8a; color:#5c4300;
    }
    .okbox {
        padding: .9rem 1rem; border-radius: 12px; background:#ecfdf3;
        border:1px solid #b8e7c9; color:#135b2d;
    }
    .small {color:#6b7280; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)



# =========================
# 拉丁名映射
# =========================
PHYLUM_LATIN_DEFAULT = {
    "蓝藻门": "Cyanobacteria (Cyanophyta)",
    "硅藻门": "Bacillariophyta",
    "金藻门": "Chrysophyta",
    "隐藻门": "Cryptophyta",
    "甲藻门": "Dinophyta (Dinoflagellata)",
    "裸藻门": "Euglenophyta",
    "绿藻门": "Chlorophyta",
}

@st.cache_data(show_spinner=False)
def load_latin_lookup() -> tuple[dict, dict]:
    """读取中文类群与拉丁名对应表。

    latin_names.csv 由已命名图片库整理而来；属级标签用 sp. 表示。
    对含 cf. 或斜杠的条目，建议后续根据正式检索表人工复核。
    """
    phylum_map = dict(PHYLUM_LATIN_DEFAULT)
    taxon_map = {}
    if LATIN_PATH.exists():
        try:
            df = pd.read_csv(LATIN_PATH)
            for _, row in df.iterrows():
                p = str(row.get("phylum", "")).strip()
                pl = str(row.get("phylum_latin", "")).strip()
                t = str(row.get("taxon", "")).strip()
                tl = str(row.get("taxon_latin", "")).strip()
                if p and pl:
                    phylum_map[p] = pl
                if t and tl and tl.lower() != "nan":
                    taxon_map[t] = tl
        except Exception:
            pass
    return phylum_map, taxon_map


def ensure_latin_columns(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    phylum_map, taxon_map = load_latin_lookup()
    if "phylum_latin" not in meta.columns:
        meta["phylum_latin"] = meta["phylum"].astype(str).map(phylum_map)
    else:
        meta["phylum_latin"] = meta["phylum_latin"].fillna("")
        missing = meta["phylum_latin"].astype(str).str.strip().eq("")
        meta.loc[missing, "phylum_latin"] = meta.loc[missing, "phylum"].astype(str).map(phylum_map)
    if "taxon_latin" not in meta.columns:
        meta["taxon_latin"] = meta["taxon"].astype(str).map(taxon_map)
    else:
        meta["taxon_latin"] = meta["taxon_latin"].fillna("")
        missing = meta["taxon_latin"].astype(str).str.strip().eq("")
        meta.loc[missing, "taxon_latin"] = meta.loc[missing, "taxon"].astype(str).map(taxon_map)
    meta["phylum_latin"] = meta["phylum_latin"].fillna("未匹配")
    meta["taxon_latin"] = meta["taxon_latin"].fillna("未匹配")
    return meta

# =========================
# 图像与特征函数
# =========================
def _to_rgb_pil(uploaded_or_bytes_or_pil) -> Image.Image:
    if isinstance(uploaded_or_bytes_or_pil, Image.Image):
        img = uploaded_or_bytes_or_pil
    else:
        img = Image.open(uploaded_or_bytes_or_pil)
    return img.convert("RGB")


def pad_image(img: Image.Image, size=(128, 128)) -> Image.Image:
    img = img.convert("RGB")
    return ImageOps.pad(
        img, size, method=Image.Resampling.LANCZOS, color=(245, 245, 245), centering=(0.5, 0.5)
    )


def foreground_mask(img: Image.Image, size=(256, 256), sensitivity: float = 0.55) -> np.ndarray:
    """显微图像的轻量前景掩膜。用于演示计数，不作为严格细胞分割。"""
    img = ImageOps.pad(img.convert("RGB"), size, method=Image.Resampling.LANCZOS, color=(245, 245, 245))
    arr = np.asarray(img).astype("float32") / 255.0
    gray = arr.mean(axis=2)
    bg = np.median(arr.reshape(-1, 3), axis=0)
    diff = np.linalg.norm(arr - bg, axis=2)
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    grad = np.sqrt(gx * gx + gy * gy)
    # sensitivity 越高，阈值越低，检出的前景越多
    q = 94 - sensitivity * 28
    thr1 = max(0.035, float(np.percentile(diff, q)))
    thr2 = max(0.015, float(np.percentile(grad, q)))
    mask = (diff > thr1) | (grad > thr2)
    # 简单去边框噪声
    mask[:3, :] = False
    mask[-3:, :] = False
    mask[:, :3] = False
    mask[:, -3:] = False
    return mask


def connected_components_count(mask: np.ndarray, min_area: int = 20, max_area_ratio: float = 0.45) -> tuple[int, list[tuple[int, int, int, int, int]]]:
    """纯 numpy/Python 连通域统计，避免 opencv 依赖。返回 count 和 bbox 列表。"""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    max_area = int(h * w * max_area_ratio)
    for y in range(h):
        xs = np.where(mask[y] & (~visited[y]))[0]
        for x0 in xs:
            if visited[y, x0] or not mask[y, x0]:
                continue
            stack = [(y, int(x0))]
            visited[y, x0] = True
            area = 0
            minx = maxx = int(x0)
            miny = maxy = int(y)
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cx < minx: minx = cx
                if cx > maxx: maxx = cx
                if cy < miny: miny = cy
                if cy > maxy: maxy = cy
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and (not visited[ny, nx]) and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if min_area <= area <= max_area:
                # 过滤细长的比例尺线条和小噪声
                bw = maxx - minx + 1
                bh = maxy - miny + 1
                aspect = max(bw / max(1, bh), bh / max(1, bw))
                if aspect < 16:
                    comps.append((minx, miny, maxx, maxy, area))
    return len(comps), comps


def overlay_mask(img: Image.Image, mask: np.ndarray) -> Image.Image:
    base = ImageOps.pad(img.convert("RGB"), mask.shape[::-1], method=Image.Resampling.LANCZOS, color=(245, 245, 245)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ov = np.array(overlay)
    ov[mask] = [255, 80, 80, 95]
    return Image.alpha_composite(base, Image.fromarray(ov)).convert("RGB")


def draw_component_boxes(img: Image.Image, comps: list[tuple[int, int, int, int, int]], size=(256, 256)) -> Image.Image:
    out = ImageOps.pad(img.convert("RGB"), size, method=Image.Resampling.LANCZOS, color=(245, 245, 245))
    draw = ImageDraw.Draw(out)
    for minx, miny, maxx, maxy, _ in comps:
        draw.rectangle([minx, miny, maxx, maxy], outline=(255, 50, 50), width=2)
    return out


def foreground_features(arr: np.ndarray) -> np.ndarray:
    gray = arr.mean(axis=2)
    med = np.median(arr.reshape(-1, 3), axis=0)
    diff = np.linalg.norm(arr - med, axis=2)
    thr = max(0.06, float(np.percentile(diff, 84)))
    mask = diff > thr
    area = mask.mean()
    if mask.any():
        ys, xs = np.where(mask)
        h = max(1, int(ys.max() - ys.min() + 1))
        w = max(1, int(xs.max() - xs.min() + 1))
        aspect = w / h
        cy = ys.mean() / mask.shape[0]
        cx = xs.mean() / mask.shape[1]
    else:
        aspect = 1.0
        cx = 0.5
        cy = 0.5
    return np.array([area, min(aspect, 10) / 10, cx, cy], dtype=np.float32)


def extract_feature_from_pil(img: Image.Image) -> np.ndarray:
    img = pad_image(img, (128, 128))
    arr = np.asarray(img).astype("float32") / 255.0
    gray = arr.mean(axis=2)
    feats = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1), density=True)
        feats.append(h.astype("float32"))
    h, _ = np.histogram(gray, bins=24, range=(0, 1), density=True)
    feats.append(h.astype("float32"))
    small = Image.fromarray((gray * 255).astype("uint8")).resize((32, 32), Image.Resampling.BILINEAR)
    sv = np.asarray(small).astype("float32") / 255.0
    sv = (sv - sv.mean()) / (sv.std() + 1e-6)
    feats.append(sv.flatten())
    row = np.interp(np.linspace(0, 127, 32), np.arange(128), gray.mean(axis=1)).astype("float32")
    col = np.interp(np.linspace(0, 127, 32), np.arange(128), gray.mean(axis=0)).astype("float32")
    feats += [row, col]
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    mag = np.sqrt(gx * gx + gy * gy)
    gh, _ = np.histogram(mag, bins=16, range=(0, float(np.percentile(mag, 99) + 1e-6)), density=True)
    feats.append(gh.astype("float32"))
    feats.append(np.array([mag.mean(), mag.std(), np.percentile(mag, 90), np.percentile(mag, 99)], dtype="float32"))
    feats.append(foreground_features(arr))
    v = np.concatenate(feats).astype("float32")
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


# =========================
# 模型加载与分类
# =========================
@st.cache_data(show_spinner=False)
def load_builtin_classifier() -> dict:
    meta = ensure_latin_columns(pd.read_csv(META_PATH))
    data = np.load(MODEL_PATH, allow_pickle=True)

    # GitHub 网页上传一次最多建议少于 100 个文件。
    # 因此紧凑部署版把 400+ 张参考图打包为 reference_images.zip。
    # 如果仓库里存在 reference_images 文件夹，也兼容直接读取文件夹。
    image_zip_bytes = None
    image_dir = REF_DIR if REF_DIR.exists() else None
    if image_dir is None and REF_ZIP.exists():
        image_zip_bytes = {}
        with zipfile.ZipFile(REF_ZIP) as zf:
            for name in zf.namelist():
                if not name.endswith("/") and Path(name).suffix.lower() in IMAGE_EXTS:
                    image_zip_bytes[Path(name).name] = zf.read(name)

    return {
        "source": "内置PDF训练库",
        "X": data["X"].astype("float32"),
        "labels": data["labels"].astype(str),
        "phyla": data["phyla"].astype(str),
        "latin_names": meta["taxon_latin"].astype(str).to_numpy(),
        "phylum_latin": meta["phylum_latin"].astype(str).to_numpy(),
        "filenames": data["filenames"].astype(str),
        "ids": data["ids"].astype(str),
        "meta": meta,
        "image_dir": image_dir,
        "image_zip_bytes": image_zip_bytes,
        "image_bytes": None,
    }


def classifier_from_zip(uploaded_zip) -> dict:
    rows = []
    feats = []
    image_bytes = []
    with zipfile.ZipFile(uploaded_zip) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS]
        for n in names:
            parts = Path(n).parts
            # 支持两种格式：属名/图片.jpg，或 门类/属名/图片.jpg
            if len(parts) >= 3 and parts[-3].endswith("门"):
                phylum = parts[-3]
                taxon = parts[-2]
            elif len(parts) >= 2:
                phylum = "自定义训练库"
                taxon = parts[-2]
            else:
                continue
            raw = zf.read(n)
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            feats.append(extract_feature_from_pil(img))
            image_bytes.append(raw)
            rows.append({
                "id": f"custom_{len(rows)+1:04d}",
                "page": "自定义",
                "image_index": len(rows) + 1,
                "phylum": phylum,
                "taxon": taxon,
                "filename": n,
            })
    if not rows:
        raise ValueError("没有在 ZIP 中读取到可用图片。请按 属名/图片.jpg 或 门类/属名/图片.jpg 的结构整理。")
    meta = ensure_latin_columns(pd.DataFrame(rows))
    return {
        "source": "上传ZIP训练库",
        "X": np.vstack(feats).astype("float32"),
        "labels": meta["taxon"].astype(str).to_numpy(),
        "phyla": meta["phylum"].astype(str).to_numpy(),
        "latin_names": meta["taxon_latin"].astype(str).to_numpy(),
        "phylum_latin": meta["phylum_latin"].astype(str).to_numpy(),
        "filenames": meta["filename"].astype(str).to_numpy(),
        "ids": meta["id"].astype(str).to_numpy(),
        "meta": meta,
        "image_dir": None,
        "image_bytes": image_bytes,
    }


def get_active_classifier() -> dict:
    if "custom_classifier" in st.session_state and st.session_state.get("use_custom", False):
        return st.session_state["custom_classifier"]
    return load_builtin_classifier()


def get_reference_image(clf: dict, idx: int) -> Image.Image:
    if clf.get("image_bytes") is not None:
        return Image.open(io.BytesIO(clf["image_bytes"][idx])).convert("RGB")
    if clf.get("image_zip_bytes") is not None:
        fname = Path(str(clf["filenames"][idx])).name
        raw = clf["image_zip_bytes"][fname]
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(clf["image_dir"] / clf["filenames"][idx]).convert("RGB")


def classify_image(img: Image.Image, clf: dict, top_m: int = 30) -> dict:
    q = extract_feature_from_pil(img)
    X = clf["X"]
    sims = X @ q
    order = np.argsort(-sims)
    top_idx = order[: min(top_m, len(order))]
    agg = defaultdict(list)
    phylum_votes = defaultdict(list)
    latin_votes = defaultdict(list)
    phylum_latin_votes = defaultdict(list)
    for idx in top_idx:
        label = str(clf["labels"][idx])
        agg[label].append(float(sims[idx]))
        phylum_votes[label].append(str(clf["phyla"][idx]))
        latin_votes[label].append(str(clf.get("latin_names", [""] * len(clf["labels"]))[idx]))
        phylum_latin_votes[label].append(str(clf.get("phylum_latin", [""] * len(clf["labels"]))[idx]))
    rows = []
    for label, vals in agg.items():
        vals_arr = np.array(vals)
        score = 0.65 * vals_arr.max() + 0.35 * vals_arr.mean()
        phylum = pd.Series(phylum_votes[label]).mode().iloc[0]
        taxon_latin = pd.Series(latin_votes[label]).mode().iloc[0] if latin_votes[label] else ""
        phylum_latin = pd.Series(phylum_latin_votes[label]).mode().iloc[0] if phylum_latin_votes[label] else ""
        rows.append({"预测属名/类群": label, "拉丁名": taxon_latin, "门类": phylum, "门类拉丁名": phylum_latin, "相似度得分": float(score), "近邻样本数": len(vals)})
    cand = pd.DataFrame(rows).sort_values("相似度得分", ascending=False).reset_index(drop=True)
    top5 = cand.head(5).copy()
    denom = float(top5["相似度得分"].clip(lower=0).sum()) + 1e-8
    top5["相对置信度"] = (top5["相似度得分"].clip(lower=0) / denom * 100).round(1)
    pred = top5.iloc[0].to_dict()
    ref_idx = top_idx[:8]
    return {"prediction": pred, "candidates": top5, "similar_indices": ref_idx, "all_sims": sims}


def classify_uploaded_file(file, clf: dict, sensitivity: float) -> dict:
    img = _to_rgb_pil(file)
    result = classify_image(img, clf)
    mask = foreground_mask(img, sensitivity=sensitivity)
    count, comps = connected_components_count(mask)
    pred = result["prediction"]
    return {
        "文件名": getattr(file, "name", "uploaded_image"),
        "预测属名/类群": pred["预测属名/类群"],
        "拉丁名": pred.get("拉丁名", ""),
        "门类": pred["门类"],
        "门类拉丁名": pred.get("门类拉丁名", ""),
        "相对置信度(%)": round(float(pred["相对置信度"]), 1),
        "前景个体数估算": int(count),
        "候选结果": result["candidates"],
        "相似样本索引": result["similar_indices"],
        "图像": img,
        "mask": mask,
        "components": comps,
    }


def download_df_button(df: pd.DataFrame, filename: str, label="下载CSV结果"):
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


# =========================
# 侧边栏
# =========================
st.sidebar.title("AI River Biology")
st.sidebar.caption("淡水浮游植物图像识别")
page = st.sidebar.radio(
    "功能导航",
    ["首页", "单张图像识别", "批量图像识别", "上传/切换训练库", "训练库概览", "方法与局限"],
)

st.sidebar.divider()
st.sidebar.subheader("模型设置")
if "use_custom" not in st.session_state:
    st.session_state["use_custom"] = False
if "custom_classifier" in st.session_state:
    st.session_state["use_custom"] = st.sidebar.toggle("使用上传的自定义训练库", value=st.session_state["use_custom"])
else:
    st.sidebar.info("当前使用内置PDF训练库")

sensitivity = st.sidebar.slider("前景分割灵敏度", 0.10, 0.90, 0.55, 0.05)
st.sidebar.caption("灵敏度越高，检出的前景越多；该计数仅用于课程原型展示。")

clf = get_active_classifier()
meta = clf["meta"]
st.sidebar.success(f"当前模型：{clf['source']}\n\n训练图像：{len(meta)} 张；标签：{meta['taxon'].nunique()} 个；拉丁名：{meta['taxon_latin'].nunique() if 'taxon_latin' in meta.columns else 0} 个")

# =========================
# 页面
# =========================
if page == "首页":
    st.markdown(
        """
        <div class="hero">
        <h1>🧫 AI River Biology：淡水浮游植物图像识别 App</h1>
        <p>上传显微镜下的淡水浮游植物图片，系统自动输出预测属名/类群、门类、相似参考图、前景个体数估算和可下载报告。</p>
        </div>
        <span class="tag">浮游植物</span><span class="tag">显微图像</span><span class="tag">属名识别</span><span class="tag">群落组成统计</span><span class="tag">CSV报告</span>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("训练图像", f"{len(meta)} 张")
    c2.metric("识别标签", f"{meta['taxon'].nunique()} 个")
    c3.metric("拉丁名", "已加入")
    c4.metric("覆盖门类", f"{meta['phylum'].nunique()} 个")
    c5.metric("算法", "特征相似度分类")

    st.subheader("这个网站能做什么")
    st.write(
        "本 App 面向 AI River Biology 课程大作业，把淡水浮游植物显微图片转换为可展示、可下载的识别结果。"
    )
    st.markdown(
        """
        1. 上传单张显微图片，输出预测中文属名/类群、拉丁名、门类和相对置信度；  
        2. 显示 Top 5 候选结果和相似参考图片，方便人工复核；  
        3. 对批量图片进行自动识别，生成群落组成统计；  
        4. 导出 CSV/TXT 结果，用于作业提交和展示；  
        5. 可上传自己的训练库 ZIP，让模型替换为你们自己的标注数据。
        """
    )
    st.markdown(
        """
        <div class="note"><b>重要说明：</b>当前版本是课程原型 App，使用你提供的已命名浮游植物图片作为内置训练库。它适合展示“图像 → 属名/门类 → 组成统计 → 报告”的完整流程；如果要做正式科研级鉴定，需要继续补充更多同一放大倍数、同一拍摄条件下的标注图片。</div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("建议展示顺序")
    st.markdown("首页流程 → 单张图片识别 → 批量识别 → 训练库概览 → 方法与局限。")

elif page == "单张图像识别":
    st.title("单张图像识别")
    st.write("上传一张淡水浮游植物显微图片，系统会给出最可能的属名/类群和相似参考图。")
    uploaded = st.file_uploader("上传图片", type=sorted([e.strip('.') for e in IMAGE_EXTS]), accept_multiple_files=False)

    # 内置测试样本选择
    with st.expander("没有测试图片？从内置训练库里选一张试试"):
        sample_row = meta.sample(1, random_state=3).iloc[0]
        label_options = meta["taxon"].value_counts().head(30).index.tolist()
        chosen_label = st.selectbox("选择一个常见类群", label_options)
        example_meta = meta[meta["taxon"] == chosen_label].head(6)
        cols = st.columns(min(6, len(example_meta)))
        for col, (idx, r) in zip(cols, example_meta.iterrows()):
            img = get_reference_image(clf, int(idx))
            col.image(img, caption=f"{r['taxon']}｜{r['phylum']}", use_container_width=True)
        st.caption("可以直接右键保存这些示例图，再回到上传框测试；紧凑版参考图存放在 reference_images.zip 中。")

    if uploaded is not None:
        try:
            res = classify_uploaded_file(uploaded, clf, sensitivity)
            pred_taxon = res["预测属名/类群"]
            pred_latin = res.get("拉丁名", "")
            pred_phylum = res["门类"]
            pred_phylum_latin = res.get("门类拉丁名", "")
            conf = res["相对置信度(%)"]
            count = res["前景个体数估算"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("预测属名/类群", pred_taxon)
            m2.metric("拉丁名", pred_latin)
            m3.metric("门类", pred_phylum)
            m4.metric("相对置信度", f"{conf}%")
            m5.metric("个体数估算", count)
            st.caption(f"门类拉丁名：{pred_phylum_latin}")

            if conf < 35:
                st.warning("相对置信度较低，建议人工复核，或补充该类群的训练图片。")
            else:
                st.success("已完成识别。建议结合相似参考图进行人工复核。")

            c1, c2, c3 = st.columns(3)
            c1.image(res["图像"], caption="原始上传图片", use_container_width=True)
            c2.image(overlay_mask(res["图像"], res["mask"]), caption="前景掩膜叠加", use_container_width=True)
            c3.image(draw_component_boxes(res["图像"], res["components"]), caption="前景连通域估算框", use_container_width=True)

            st.subheader("Top 5 候选结果")
            st.dataframe(res["候选结果"], use_container_width=True, hide_index=True)
            chart_df = res["候选结果"][["预测属名/类群", "相对置信度"]].set_index("预测属名/类群")
            st.bar_chart(chart_df)

            st.subheader("相似参考图片")
            cols = st.columns(4)
            for j, idx in enumerate(res["相似样本索引"][:8]):
                rimg = get_reference_image(clf, int(idx))
                cap = f"{clf['labels'][idx]}｜{clf.get('latin_names', [''] * len(clf['labels']))[idx]}｜{clf['phyla'][idx]}"
                cols[j % 4].image(rimg, caption=cap, use_container_width=True)

            report = pd.DataFrame([{k: v for k, v in res.items() if k not in ["候选结果", "相似样本索引", "图像", "mask", "components"]}])
            st.subheader("结果下载")
            download_df_button(report, "phytoplankton_single_result.csv")
            txt = (
                f"淡水浮游植物图像识别报告\n"
                f"文件名：{res['文件名']}\n"
                f"预测属名/类群：{pred_taxon}\n"
                f"拉丁名：{pred_latin}\n"
                f"门类：{pred_phylum}\n"
                f"门类拉丁名：{pred_phylum_latin}\n"
                f"相对置信度：{conf}%\n"
                f"前景个体数估算：{count}\n"
                f"说明：该结果由课程原型模型自动生成，应结合显微形态特征进行人工复核。\n"
            )
            st.download_button("下载TXT报告", txt.encode("utf-8-sig"), "phytoplankton_single_report.txt", "text/plain")
        except Exception as e:
            st.error(f"识别失败：{e}")

elif page == "批量图像识别":
    st.title("批量图像识别")
    st.write("一次上传多张浮游植物显微图片，自动生成属名预测、门类统计和群落组成表。")
    files = st.file_uploader("上传多张图片", type=sorted([e.strip('.') for e in IMAGE_EXTS]), accept_multiple_files=True)
    if files:
        rows = []
        progress = st.progress(0)
        for i, f in enumerate(files):
            try:
                res = classify_uploaded_file(f, clf, sensitivity)
                rows.append({k: v for k, v in res.items() if k not in ["候选结果", "相似样本索引", "图像", "mask", "components"]})
            except Exception as e:
                rows.append({"文件名": getattr(f, "name", "unknown"), "错误": str(e)})
            progress.progress((i + 1) / len(files))
        df = pd.DataFrame(rows)
        st.subheader("批量识别结果")
        st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "phytoplankton_batch_results.csv")

        if "预测属名/类群" in df.columns:
            st.subheader("群落组成统计")
            c1, c2 = st.columns(2)
            with c1:
                st.write("按门类统计")
                phylum_count = df["门类"].value_counts().rename_axis("门类").reset_index(name="数量")
                st.dataframe(phylum_count, hide_index=True, use_container_width=True)
                st.bar_chart(phylum_count.set_index("门类")["图片数"])
            with c2:
                st.write("按属名/类群统计")
                taxon_count = df["预测属名/类群"].value_counts().rename_axis("属名/类群").reset_index(name="数量")
                st.dataframe(taxon_count, hide_index=True, use_container_width=True)
                st.bar_chart(taxon_count.set_index("属名/类群").head(20))

elif page == "上传/切换训练库":
    st.title("上传/切换训练库")
    st.write("你可以继续使用内置训练库，也可以上传自己的标注图片 ZIP，让 App 按你的图片重新训练。")
    st.markdown(
        """
        推荐 ZIP 结构：
        ```text
        phytoplankton_dataset.zip
        ├── 硅藻门/
        │   ├── 小环藻属/
        │   │   ├── img001.jpg
        │   │   └── img002.jpg
        │   └── 舟形藻属/
        ├── 绿藻门/
        │   └── 衣藻属/
        └── 蓝藻门/
            └── 伪鱼腥藻属/
        ```
        也支持简化结构：`小环藻属/img001.jpg`。
        """
    )
    upzip = st.file_uploader("上传训练库 ZIP", type=["zip"])
    if upzip is not None:
        try:
            with st.spinner("正在读取图片并构建特征库……"):
                custom = classifier_from_zip(upzip)
            st.session_state["custom_classifier"] = custom
            st.session_state["use_custom"] = True
            st.success(f"自定义训练库已加载：{len(custom['meta'])} 张图片，{custom['meta']['taxon'].nunique()} 个标签。")
            st.dataframe(custom["meta"].head(20), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"训练库读取失败：{e}")
    st.divider()
    if st.button("切回内置PDF训练库"):
        st.session_state["use_custom"] = False
        st.success("已切回内置PDF训练库。")

elif page == "训练库概览":
    st.title("训练库概览")
    st.write(f"当前训练库来源：**{clf['source']}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("训练图片", len(meta))
    c2.metric("标签数量", meta["taxon"].nunique())
    c3.metric("拉丁名数量", meta["taxon_latin"].nunique() if "taxon_latin" in meta.columns else 0)
    c4.metric("门类数量", meta["phylum"].nunique())

    st.subheader("按门类统计")
    phylum_count = meta.groupby(["phylum", "phylum_latin"]).size().reset_index(name="图片数").rename(columns={"phylum":"门类", "phylum_latin":"门类拉丁名"})
    st.dataframe(phylum_count, hide_index=True, use_container_width=True)
    st.bar_chart(phylum_count.set_index("门类")["图片数"])

    st.subheader("按属名/类群统计（前30个）")
    taxon_count = meta.groupby(["taxon", "taxon_latin"]).size().reset_index(name="图片数").rename(columns={"taxon":"属名/类群", "taxon_latin":"拉丁名"}).sort_values("图片数", ascending=False)
    st.dataframe(taxon_count.head(30), hide_index=True, use_container_width=True)
    st.bar_chart(taxon_count.set_index("属名/类群")["图片数"].head(30))

    st.subheader("浏览参考图片")
    labels = meta["taxon"].value_counts().index.tolist()
    chosen = st.selectbox("选择属名/类群", labels)
    subset = meta[meta["taxon"] == chosen].head(12)
    cols = st.columns(4)
    for j, (idx, r) in enumerate(subset.iterrows()):
        # meta index 与内置 npz 顺序一致；自定义也一致
        img = get_reference_image(clf, int(idx))
        cols[j % 4].image(img, caption=f"{r['taxon']}｜{r.get('taxon_latin', '')}｜{r['phylum']}", use_container_width=True)

elif page == "方法与局限":
    st.title("方法与局限")
    st.subheader("算法流程")
    st.markdown(
        """
        本 App 采用轻量化的课程原型流程：

        **显微图像输入 → 图像标准化 → 颜色/灰度/纹理/边缘/前景形态特征提取 → 与训练库图片相似度匹配 → Top 候选中文属名/拉丁名 → 批量统计与报告导出。**

        它不是大型深度学习模型，也不是严格的形态分类检索表。优点是部署轻、可解释、可直接用你提供的“图片—属名”材料训练；缺点是对拍摄条件、背景颜色、比例尺、焦平面和同属内部形态差异比较敏感。
        """
    )
    st.subheader("为什么这里不用 YOLOv8")
    st.write(
        "浮游植物显微图像如果要用 YOLOv8，需要逐个细胞画检测框，并给每个框标注属名。你现在提供的是已命名图像页，更适合先做图像分类/相似识别。后续如果你愿意逐个细胞标框，可以再升级成 YOLOv8 目标检测版本。"
    )
    st.subheader("目前内置库的适用范围")
    st.markdown(
        f"""
        - 内置训练库图像数：**{len(load_builtin_classifier()['meta'])} 张**；  
        - 覆盖标签数：**{load_builtin_classifier()['meta']['taxon'].nunique()} 个属名/类群**，并加入对应拉丁名；  
        - 覆盖门类：蓝藻门、硅藻门、金藻门、隐藻门、甲藻门、裸藻门、绿藻门；  
        - 更适合做课程展示和原型验证，不建议作为正式鉴定结论直接使用。
        """
    )
    st.subheader("展示时可以这样说")
    st.markdown(
        """
        > 本网站基于已命名淡水浮游植物显微图片构建参考训练库，覆盖多个淡水浮游植物门类和常见属名。用户上传显微图片后，系统提取颜色、纹理、边缘和形态特征，与训练库进行相似度匹配，输出最可能的中文属名/类群、拉丁名、门类、置信度和群落组成统计。该系统适合用于 AI River Biology 中的 bioimaging 和 target identification 场景，并可通过继续补充标注图片提高识别能力。
        """
    )
