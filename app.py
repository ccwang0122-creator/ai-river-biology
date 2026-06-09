"""
AI River Biology - Fish & Benthic Macroinvertebrate Recognition App
Course prototype for River Dynamics and Integrated River Management.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from scipy import ndimage

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample_data"

st.set_page_config(
    page_title="AI River Biology",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Taxon library
# ---------------------------
TAXON_LIBRARY: List[Dict[str, object]] = [
    {
        "taxon": "Cyprinidae fish",
        "common_name": "Carp / minnow type fish",
        "category": "Fish",
        "prototype": [0.18, 4.1, 0.43, 0.22, 0.07, 0.46, 0.47, 0.39, 0.16],
        "keywords": ["fish", "carp", "minnow", "cyprin", "鲤", "鱼", "鲫", "麦穗鱼"],
        "traits": "Elongated body, obvious head-tail axis, relatively smooth edge.",
        "indicator": "Mobile nekton; useful for river connectivity and habitat continuity assessment.",
        "habitat": "Pools, runs and near-bank habitats with connected flow paths.",
        "management": "Presence of multiple size classes may indicate connected habitat and recruitment potential.",
    },
    {
        "taxon": "Cobitidae / small benthic fish",
        "common_name": "Loach type fish",
        "category": "Fish",
        "prototype": [0.12, 5.2, 0.35, 0.18, 0.10, 0.42, 0.40, 0.34, 0.18],
        "keywords": ["loach", "cobit", "泥鳅", "鳅"],
        "traits": "Very slender body, benthic posture, smooth outline.",
        "indicator": "Often linked with shallow, low-velocity and heterogeneous substrate habitats.",
        "habitat": "Sandy or muddy river margins and slow-flow microhabitats.",
        "management": "Can support interpretation of local habitat heterogeneity and bed substrate condition.",
    },
    {
        "taxon": "Ephemeroptera nymph",
        "common_name": "Mayfly nymph",
        "category": "Benthic macroinvertebrate",
        "prototype": [0.08, 2.9, 0.30, 0.16, 0.28, 0.48, 0.45, 0.35, 0.22],
        "keywords": ["mayfly", "ephemeroptera", "蜉蝣", "蜉蝣目"],
        "traits": "Flattened or slender insect nymph, visible legs, often with tail filaments.",
        "indicator": "Generally sensitive; commonly used in biological water quality assessment.",
        "habitat": "Riffles, gravel-cobble substrate, oxygenated water.",
        "management": "High relative occurrence may indicate better habitat quality and oxygen conditions.",
    },
    {
        "taxon": "Plecoptera nymph",
        "common_name": "Stonefly nymph",
        "category": "Benthic macroinvertebrate",
        "prototype": [0.07, 2.7, 0.28, 0.15, 0.32, 0.36, 0.34, 0.29, 0.24],
        "keywords": ["stonefly", "plecoptera", "襀翅", "石蝇"],
        "traits": "Elongated nymph with two tail filaments and clear legs.",
        "indicator": "Highly sensitive group; often associated with cold, oxygen-rich streams.",
        "habitat": "Coarse substrate, riffle habitats, high dissolved oxygen.",
        "management": "Occurrence can be used as evidence for high-quality benthic habitat.",
    },
    {
        "taxon": "Trichoptera larva",
        "common_name": "Caddisfly larva",
        "category": "Benthic macroinvertebrate",
        "prototype": [0.10, 2.0, 0.38, 0.24, 0.24, 0.52, 0.49, 0.38, 0.21],
        "keywords": ["caddis", "trichoptera", "毛翅", "石蚕"],
        "traits": "Larval body or case-bearing form; case may appear as rough tube or grains.",
        "indicator": "Often related to substrate composition and organic matter processing.",
        "habitat": "Stones, woody debris and leaf packs in flowing waters.",
        "management": "Useful for diagnosing substrate stability and detrital habitat quality.",
    },
    {
        "taxon": "Chironomidae larva",
        "common_name": "Non-biting midge larva",
        "category": "Benthic macroinvertebrate",
        "prototype": [0.05, 6.8, 0.22, 0.10, 0.22, 0.58, 0.32, 0.28, 0.20],
        "keywords": ["chironom", "midge", "摇蚊", "红虫"],
        "traits": "Small worm-like larva, often red or brown, no obvious hard shell.",
        "indicator": "Tolerant group; high dominance can suggest organic enrichment or fine sediment.",
        "habitat": "Fine sediment, depositional zones, macrophyte beds.",
        "management": "Dominance should be interpreted with water-quality and sediment information.",
    },
    {
        "taxon": "Oligochaeta",
        "common_name": "Aquatic worm",
        "category": "Benthic macroinvertebrate",
        "prototype": [0.04, 8.5, 0.18, 0.08, 0.18, 0.55, 0.38, 0.34, 0.16],
        "keywords": ["oligochaeta", "worm", "tubifex", "寡毛", "水蚯蚓", "蚯蚓"],
        "traits": "Long soft body, very high length-width ratio.",
        "indicator": "Often tolerant of fine sediment and organic matter accumulation.",
        "habitat": "Fine sediment, slow-flow areas and organically enriched substrates.",
        "management": "Useful for screening possible sedimentation or organic pollution pressure.",
    },
    {
        "taxon": "Gastropoda",
        "common_name": "Freshwater snail",
        "category": "Benthic macroinvertebrate",
        "prototype": [0.10, 1.3, 0.58, 0.50, 0.16, 0.45, 0.40, 0.32, 0.18],
        "keywords": ["snail", "gastropod", "螺", "螺类"],
        "traits": "Compact shell outline, low length-width ratio, relatively high compactness.",
        "indicator": "Grazing and periphyton-related group; response depends on water quality and substrate.",
        "habitat": "Stones, macrophytes, margins and slow-flow habitats.",
        "management": "Can be interpreted together with periphyton and substrate condition.",
    },
    {
        "taxon": "Bivalvia",
        "common_name": "Freshwater mussel / clam",
        "category": "Benthic macroinvertebrate",
        "prototype": [0.13, 1.8, 0.62, 0.55, 0.12, 0.43, 0.39, 0.32, 0.15],
        "keywords": ["bivalve", "mussel", "clam", "unionid", "蚌", "贝", "双壳"],
        "traits": "Oval shell, compact outline, hard shell texture.",
        "indicator": "Filter-feeding group; sensitive to substrate stability and hydrological connectivity.",
        "habitat": "Stable sand-gravel substrate, margins and low-velocity zones.",
        "management": "Useful for linking bed stability, water quality and filter-feeding ecological function.",
    },
    {
        "taxon": "Limnoperna fortunei",
        "common_name": "Golden mussel / freshwater mytilid",
        "category": "Benthic macroinvertebrate",
        "prototype": [0.09, 1.9, 0.57, 0.48, 0.15, 0.52, 0.46, 0.28, 0.18],
        "keywords": ["limnoperna", "fortunei", "golden mussel", "freshwater mytilid", "淡水壳菜", "壳菜"],
        "traits": "Small triangular to oval shell, often attached to hard substrate by byssus.",
        "indicator": "Invasive filter feeder; early detection is important for river and water-transfer management.",
        "habitat": "Hard substrates, intake structures, canal walls and stable surfaces.",
        "management": "Flag for rapid reporting, density monitoring and infrastructure biofouling risk assessment.",
    },
]

LIB_DF = pd.DataFrame(
    [
        {
            "Taxon": item["taxon"],
            "Common name": item["common_name"],
            "Category": item["category"],
            "Key traits": item["traits"],
            "Indicator meaning": item["indicator"],
            "Habitat": item["habitat"],
        }
        for item in TAXON_LIBRARY
    ]
)

FEATURE_NAMES = [
    "area_ratio",
    "aspect_ratio",
    "extent",
    "compactness",
    "edge_density",
    "mean_r",
    "mean_g",
    "mean_b",
    "gray_std",
]
FEATURE_SCALE = np.array([0.16, 4.0, 0.35, 0.30, 0.18, 0.20, 0.20, 0.20, 0.12], dtype=float)


# ---------------------------
# Styling
# ---------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        .hero {
            padding: 1.35rem 1.55rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #EAF7FF 0%, #F6FBF8 52%, #FFF7EA 100%);
            border: 1px solid rgba(28, 98, 148, 0.15);
            margin-bottom: 1rem;
        }
        .hero h1 {margin-bottom: .35rem; font-size: 2.15rem; color: #103A5C;}
        .hero p {font-size: 1.02rem; color: #475569; margin-bottom: 0;}
        .pill {
            display: inline-block; padding: .25rem .65rem; margin: .15rem .2rem .15rem 0;
            border-radius: 999px; background: #EEF6FF; border: 1px solid #C8E4FF; color: #174E79; font-size: .82rem;
        }
        .small-note {font-size:.86rem; color:#64748B;}
        .card {
            border: 1px solid #E2E8F0; border-radius: 18px; padding: 1rem; background: #FFFFFF;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04); min-height: 112px;
        }
        .warning-card {
            border: 1px solid #FBD38D; border-radius: 14px; padding: .85rem; background:#FFFBEB; color:#7C4A03;
        }
        .success-card {
            border: 1px solid #BBF7D0; border-radius: 14px; padding: .85rem; background:#F0FDF4; color:#14532D;
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #E2E8F0; padding: .8rem; border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# Image utilities and demo images
# ---------------------------
def ensure_demo_images() -> None:
    """Create small synthetic demo images if they are absent."""
    SAMPLE_DIR.mkdir(exist_ok=True)
    demos = {
        "demo_fish.png": "fish",
        "demo_mayfly.png": "mayfly",
        "demo_mussel.png": "mussel",
    }
    for filename, kind in demos.items():
        path = SAMPLE_DIR / filename
        if path.exists():
            continue
        img = Image.new("RGB", (640, 420), "white")
        draw = ImageDraw.Draw(img)
        if kind == "fish":
            # body
            draw.ellipse((125, 150, 470, 270), fill=(118, 151, 129), outline=(55, 84, 77), width=5)
            # tail
            draw.polygon([(470, 210), (575, 130), (548, 210), (575, 290)], fill=(95, 135, 124), outline=(55, 84, 77))
            # head/eye/fins
            draw.ellipse((170, 185, 190, 205), fill=(20, 31, 36))
            draw.polygon([(280, 155), (345, 82), (365, 168)], fill=(90, 123, 112), outline=(55, 84, 77))
            draw.polygon([(320, 265), (375, 335), (390, 255)], fill=(90, 123, 112), outline=(55, 84, 77))
            draw.line((210, 215, 455, 212), fill=(58, 92, 88), width=3)
        elif kind == "mayfly":
            draw.ellipse((250, 115, 380, 310), fill=(145, 124, 91), outline=(69, 55, 42), width=4)
            draw.ellipse((270, 75, 360, 145), fill=(128, 109, 82), outline=(69, 55, 42), width=4)
            for x1, y1, x2, y2 in [(260, 165, 140, 90), (275, 210, 120, 210), (285, 255, 150, 330), (370, 165, 500, 90), (360, 210, 520, 210), (350, 255, 490, 330)]:
                draw.line((x1, y1, x2, y2), fill=(50, 45, 40), width=5)
            for x2 in [235, 320, 405]:
                draw.line((315, 310, x2, 395), fill=(50, 45, 40), width=4)
        else:
            draw.ellipse((210, 120, 430, 320), fill=(178, 143, 83), outline=(84, 60, 38), width=5)
            draw.arc((210, 120, 430, 320), 210, 25, fill=(90, 65, 40), width=5)
            draw.arc((250, 155, 395, 285), 205, 20, fill=(120, 88, 48), width=4)
            draw.line((232, 260, 405, 170), fill=(100, 72, 38), width=3)
        img.save(path)


def load_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file).convert("RGB")
    return ImageOps.exif_transpose(image)


def resize_for_processing(img: Image.Image, max_side: int = 900) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img.copy()
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)))


def otsu_threshold(gray: np.ndarray) -> float:
    values = np.clip(gray.ravel(), 0, 1)
    hist, edges = np.histogram(values, bins=256, range=(0, 1))
    hist = hist.astype(float)
    prob = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = np.nan
    sigma = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma))
    return float(centers[idx])


def rgb_to_hsv_simple(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=2)
    mn = np.min(arr, axis=2)
    diff = mx - mn
    hue = np.zeros_like(mx)
    mask = diff > 1e-8
    rmask = mask & (mx == r)
    gmask = mask & (mx == g)
    bmask = mask & (mx == b)
    hue[rmask] = ((g[rmask] - b[rmask]) / diff[rmask]) % 6
    hue[gmask] = ((b[gmask] - r[gmask]) / diff[gmask]) + 2
    hue[bmask] = ((r[bmask] - g[bmask]) / diff[bmask]) + 4
    hue = hue / 6.0
    sat = np.where(mx == 0, 0, diff / np.maximum(mx, 1e-8))
    return hue, sat, mx


def segment_foreground(img: Image.Image, sensitivity: float = 0.55) -> np.ndarray:
    """Return a foreground mask for likely organisms.

    The method is transparent and intentionally light-weight for course demos:
    brightness thresholding + saturation cue + morphological cleanup.
    """
    img = resize_for_processing(img)
    arr = np.asarray(img).astype(float) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    _, sat, val = rgb_to_hsv_simple(arr)

    t = otsu_threshold(gray)
    bg_is_bright = float(np.mean(gray)) > 0.52
    margin = 0.03 + 0.10 * sensitivity

    if bg_is_bright:
        mask_gray = gray < min(t + margin, 0.92)
    else:
        mask_gray = np.abs(gray - np.median(gray)) > (0.10 - 0.04 * sensitivity)

    sat_thr = np.percentile(sat, 65) * (0.75 - 0.20 * sensitivity)
    mask_sat = (sat > max(sat_thr, 0.10)) & (val < 0.96)

    mask = mask_gray | mask_sat
    # Remove image borders, text-like dust and fill holes.
    h, w = mask.shape
    border = max(2, int(min(h, w) * 0.01))
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)

    labels, num = ndimage.label(mask)
    if num > 0:
        areas = np.bincount(labels.ravel())
        min_area = max(35, int(0.00035 * mask.size))
        keep = np.zeros(num + 1, dtype=bool)
        keep[np.where(areas >= min_area)[0]] = True
        keep[0] = False
        mask = keep[labels]

    # If the automatic mask is still unrealistic, switch to conservative edge/contrast mask.
    area = float(mask.mean())
    if area < 0.002 or area > 0.70:
        contrast = np.abs(gray - np.median(gray))
        mask = contrast > np.percentile(contrast, 85)
        mask = ndimage.binary_closing(mask, structure=np.ones((4, 4)))
        mask = ndimage.binary_fill_holes(mask)

    return mask.astype(bool)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def overlay_mask(img: Image.Image, mask: np.ndarray) -> Image.Image:
    base = resize_for_processing(img).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask.astype(np.uint8) * 120), mode="L").resize(base.size)
    # Cyan overlay for foreground.
    color = Image.new("RGBA", base.size, (20, 150, 170, 0))
    color.putalpha(mask_img)
    return Image.alpha_composite(base, color).convert("RGB")


def connected_component_stats(mask: np.ndarray) -> Tuple[int, int, Optional[Tuple[int, int, int, int]], np.ndarray]:
    labels, n = ndimage.label(mask)
    if n == 0:
        return 0, 0, None, labels
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    min_area = max(60, int(0.0005 * mask.size))
    valid = np.where(areas >= min_area)[0]
    if len(valid) == 0:
        return 0, 0, None, labels
    largest = int(valid[np.argmax(areas[valid])])
    positions = np.argwhere(labels == largest)
    y0, x0 = positions.min(axis=0)
    y1, x1 = positions.max(axis=0) + 1
    return int(len(valid)), largest, (int(x0), int(y0), int(x1), int(y1)), labels


def extract_features(img: Image.Image, sensitivity: float = 0.55) -> Tuple[Dict[str, float], np.ndarray]:
    proc = resize_for_processing(img)
    arr = np.asarray(proc).astype(float) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mask = segment_foreground(proc, sensitivity=sensitivity)
    count, largest, bbox, labels = connected_component_stats(mask)

    if largest > 0:
        obj = labels == largest
    else:
        obj = mask

    area = float(obj.sum())
    total = float(mask.size)
    area_ratio = area / max(total, 1.0)

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        bw = max(x1 - x0, 1)
        bh = max(y1 - y0, 1)
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        extent = area / max(bw * bh, 1)
    else:
        aspect_ratio = 1.0
        extent = 0.0

    eroded = ndimage.binary_erosion(obj, structure=np.ones((3, 3))) if obj.any() else obj
    perimeter = float(np.logical_xor(obj, eroded).sum())
    compactness = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    edge_density = perimeter / max(area, 1.0)

    if obj.any():
        foreground = arr[obj]
        mean_rgb = foreground.mean(axis=0)
        gray_std = float(gray[obj].std())
    else:
        mean_rgb = np.array([0.5, 0.5, 0.5])
        gray_std = 0.0

    features = {
        "area_ratio": float(area_ratio),
        "aspect_ratio": float(aspect_ratio),
        "extent": float(extent),
        "compactness": float(compactness),
        "edge_density": float(edge_density),
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "gray_std": float(gray_std),
        "count": float(count),
    }
    return features, mask


def vector_from_features(features: Dict[str, float]) -> np.ndarray:
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=float)


# ---------------------------
# Classifiers
# ---------------------------
def filename_keyword_score(filename: str, item: Dict[str, object]) -> float:
    name = filename.lower()
    score = 0.0
    for kw in item.get("keywords", []):
        kw = str(kw).lower()
        if kw and kw in name:
            score += 0.45
    return min(score, 0.75)


def heuristic_boost(features: Dict[str, float], item: Dict[str, object]) -> float:
    taxon = str(item["taxon"]).lower()
    category = str(item["category"]).lower()
    aspect = features["aspect_ratio"]
    compact = features["compactness"]
    edge = features["edge_density"]
    extent = features["extent"]
    area = features["area_ratio"]

    boost = 0.0
    if category == "fish" and aspect > 3.0 and extent > 0.25 and edge < 0.24:
        boost += 0.18
    if "loach" in str(item["common_name"]).lower() and aspect > 4.8:
        boost += 0.15
    if "oligochaeta" in taxon and aspect > 6.0 and compact < 0.18:
        boost += 0.18
    if "chironomidae" in taxon and aspect > 4.8 and area < 0.09:
        boost += 0.12
    if "gastropoda" in taxon and aspect < 1.7 and compact > 0.28:
        boost += 0.18
    if "bivalvia" in taxon and aspect < 2.4 and compact > 0.28:
        boost += 0.14
    if "limnoperna" in taxon and aspect < 2.4 and 0.08 <= area <= 0.20:
        boost += 0.10
    if any(x in taxon for x in ["ephemeroptera", "plecoptera", "trichoptera"]) and edge > 0.20:
        boost += 0.14
    return boost


def classify_library(features: Dict[str, float], filename: str = "") -> pd.DataFrame:
    vec = vector_from_features(features)
    scores = []
    for item in TAXON_LIBRARY:
        proto = np.array(item["prototype"], dtype=float)
        dist = np.sqrt(np.sum(((vec - proto) / FEATURE_SCALE) ** 2))
        base = 1.0 / (1.0 + dist)
        score = base + filename_keyword_score(filename, item) + heuristic_boost(features, item)
        scores.append(score)
    scores = np.array(scores, dtype=float)
    # Softmax-like confidence from transparent prototype scores
    exp_scores = np.exp((scores - scores.max()) * 3.0)
    conf = exp_scores / exp_scores.sum()
    rows = []
    for item, s, c in zip(TAXON_LIBRARY, scores, conf):
        rows.append(
            {
                "Predicted taxon": item["taxon"],
                "Common name": item["common_name"],
                "Category": item["category"],
                "Confidence": round(float(c), 3),
                "Model score": round(float(s), 3),
                "Key traits used": item["traits"],
                "Ecological interpretation": item["indicator"],
                "Management note": item["management"],
            }
        )
    return pd.DataFrame(rows).sort_values("Confidence", ascending=False).reset_index(drop=True)


def build_reference_database(zip_file, sensitivity: float = 0.55) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_file) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if not member.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")):
                continue
            parts = Path(member).parts
            if len(parts) >= 2:
                label = parts[-2]
            else:
                label = "Unknown"
            with zf.open(member) as fp:
                img = Image.open(fp).convert("RGB")
                img = ImageOps.exif_transpose(img)
                features, _ = extract_features(img, sensitivity=sensitivity)
                row = {"label": label, "file": member}
                row.update({name: features[name] for name in FEATURE_NAMES})
                rows.append(row)
    return pd.DataFrame(rows)


def classify_reference(features: Dict[str, float], reference_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    if reference_df is None or reference_df.empty:
        return pd.DataFrame()
    vec = vector_from_features(features)
    ref = reference_df[FEATURE_NAMES].to_numpy(dtype=float)
    dist = np.sqrt(np.sum(((ref - vec) / FEATURE_SCALE) ** 2, axis=1))
    temp = reference_df.copy()
    temp["distance"] = dist
    temp = temp.sort_values("distance").head(max(k, 1))
    label_scores = []
    for label, group in temp.groupby("label"):
        score = float(np.sum(1 / (1 + group["distance"].to_numpy())))
        nearest_file = str(group.iloc[0]["file"])
        label_scores.append({"Predicted taxon": label, "Nearest reference": nearest_file, "score": score})
    out = pd.DataFrame(label_scores).sort_values("score", ascending=False).reset_index(drop=True)
    if not out.empty:
        out["Confidence"] = out["score"] / out["score"].sum()
        out["Confidence"] = out["Confidence"].round(3)
        out = out.drop(columns=["score"])
    return out


# ---------------------------
# Reporting helpers
# ---------------------------
def make_result_record(
    filename: str,
    features: Dict[str, float],
    prediction_df: pd.DataFrame,
    site: str,
    sample_date: str,
    latitude: Optional[float],
    longitude: Optional[float],
    water_note: str,
) -> Dict[str, object]:
    top = prediction_df.iloc[0].to_dict() if not prediction_df.empty else {}
    record = {
        "file": filename,
        "site": site,
        "date": sample_date,
        "latitude": latitude,
        "longitude": longitude,
        "predicted_taxon": top.get("Predicted taxon", "Unknown"),
        "category": top.get("Category", "Unknown"),
        "confidence": top.get("Confidence", 0),
        "detected_count": int(features.get("count", 0)),
        "water_note": water_note,
    }
    for name in FEATURE_NAMES:
        record[name] = round(float(features.get(name, 0)), 4)
    return record


def ecological_grade(top_taxon: str, category: str) -> Tuple[str, str]:
    t = top_taxon.lower()
    if "plecoptera" in t or "ephemeroptera" in t:
        return "Sensitive / high-value bioindicator", "Prioritize habitat protection and check riffle-substrate integrity."
    if "trichoptera" in t:
        return "Moderately sensitive bioindicator", "Interpret together with substrate stability and organic matter."
    if "chironomidae" in t or "oligochaeta" in t:
        return "Tolerant bioindicator", "Check fine sediment, organic enrichment and local dissolved oxygen."
    if "limnoperna" in t:
        return "Invasive biofouling alert", "Recommend rapid verification, density survey and infrastructure risk screening."
    if category == "Fish":
        return "Connectivity and habitat-use indicator", "Combine with river connectivity, flow refuge and bank habitat information."
    return "General benthic indicator", "Use as supporting evidence in multi-metric biological assessment."


def report_text(record: Dict[str, object], prediction_df: pd.DataFrame) -> str:
    top = prediction_df.iloc[0].to_dict() if not prediction_df.empty else {}
    grade, action = ecological_grade(str(record.get("predicted_taxon", "")), str(record.get("category", "")))
    top3 = prediction_df.head(3)[["Predicted taxon", "Confidence"]].to_string(index=False)
    return f"""AI River Biology - Automatic Recognition Report

Sample information
- File: {record.get('file')}
- Site: {record.get('site')}
- Date: {record.get('date')}
- Location: {record.get('latitude')}, {record.get('longitude')}
- Water-quality / habitat note: {record.get('water_note')}

Recognition result
- Predicted taxon: {record.get('predicted_taxon')}
- Category: {record.get('category')}
- Confidence: {record.get('confidence')}
- Detected individual count: {record.get('detected_count')}
- Bioindicator type: {grade}
- Suggested management interpretation: {action}

Top-3 candidates
{top3}

Extracted image features
- Area ratio: {record.get('area_ratio')}
- Aspect ratio: {record.get('aspect_ratio')}
- Extent: {record.get('extent')}
- Compactness: {record.get('compactness')}
- Edge density: {record.get('edge_density')}

Note: This course prototype uses transparent segmentation, morphology and reference-library matching. For formal monitoring, results should be verified by taxonomic experts and improved with labelled training images.
"""


def dataframe_download(df: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def show_workflow_cards() -> None:
    cols = st.columns(5)
    labels = [
        ("1", "Input images", "Fish or benthic macroinvertebrate photos from field sampling."),
        ("2", "Preprocess", "Resize, brightness normalization and foreground segmentation."),
        ("3", "Feature extraction", "Shape, color, texture and count-related indicators."),
        ("4", "AI recognition", "Prototype matching or custom reference-library kNN."),
        ("5", "Output", "Species/taxon, count, location, report and CSV download."),
    ]
    for col, (num, title, desc) in zip(cols, labels):
        with col:
            st.markdown(f"<div class='card'><b>{num}. {title}</b><br><span class='small-note'>{desc}</span></div>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class='hero'>
            <h1>🐟 AI River Biology</h1>
            <p><b>Automatic recognition of fish and benthic macroinvertebrates</b> from river bioimages. The app turns field photos into taxon, count, location and ecological interpretation outputs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='pill'>Bioimaging</span><span class='pill'>Target identification</span><span class='pill'>Fish</span><span class='pill'>Benthic macroinvertebrates</span><span class='pill'>CSV report</span>",
        unsafe_allow_html=True,
    )


# ---------------------------
# Pages
# ---------------------------
def page_home() -> None:
    render_header()
    st.write("")
    show_workflow_cards()
    st.write("")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("What this website does")
        st.markdown(
            """
            This app is designed for an **AI River Biology** assignment. It accepts biological images collected during river surveys and produces:

            - **Predicted taxon / species group** for fish and benthic macroinvertebrates;
            - **Individual count estimate** based on segmented foreground objects;
            - **Sampling site, date and location record**;
            - **Ecological interpretation** for river management;
            - **Downloadable CSV and text report** for final submission.
            """
        )
        st.markdown(
            "<div class='warning-card'><b>Prototype note:</b> The built-in model is a transparent feature-matching classifier for course demonstration. It becomes much stronger when you upload a labelled reference image ZIP from your own field trip.</div>",
            unsafe_allow_html=True,
        )
    with right:
        ensure_demo_images()
        demo_cols = st.columns(3)
        for c, name in zip(demo_cols, ["demo_fish.png", "demo_mayfly.png", "demo_mussel.png"]):
            with c:
                st.image(str(SAMPLE_DIR / name), caption=name.replace("demo_", "").replace(".png", ""), use_container_width=True)

    st.subheader("Recommended submission structure")
    st.markdown(
        """
        1. Show the website homepage and workflow.
        2. Upload one fish photo and one benthic animal photo.
        3. Present predicted taxon, confidence, count and ecological interpretation.
        4. Demonstrate batch recognition and CSV download.
        5. Explain limitations and future improvement using labelled images and deep learning.
        """
    )


def page_single(sensitivity: float, reference_df: Optional[pd.DataFrame]) -> None:
    st.header("Single-image recognition")
    st.caption("Upload one fish or benthic macroinvertebrate image. The app will segment the organism, extract features and return top taxon candidates.")

    col_meta, col_upload = st.columns([0.92, 1.2])
    with col_meta:
        st.subheader("Sample metadata")
        site = st.text_input("Sampling site / river reach", value="River field site A")
        sample_date = st.date_input("Sampling date", value=date.today())
        lat = st.number_input("Latitude", value=39.9042, format="%.6f")
        lon = st.number_input("Longitude", value=116.4074, format="%.6f")
        method = st.selectbox("Collection method", ["Kick sampling", "D-net", "Electrofishing", "Visual photo record", "Other"])
        wt = st.number_input("Water temperature (°C)", value=18.0, step=0.5)
        do = st.number_input("Dissolved oxygen (mg/L)", value=7.5, step=0.1)
        habitat = st.text_area("Habitat / water-quality note", value=f"Method: {method}; gravel-cobble substrate; WT={wt} °C; DO={do} mg/L.")

    with col_upload:
        st.subheader("Image input")
        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff", "webp"])
        demo_choice = st.selectbox("Or use built-in demo image", ["None", "demo_fish.png", "demo_mayfly.png", "demo_mussel.png"])

    image = None
    filename = ""
    if uploaded is not None:
        image = load_image(uploaded)
        filename = uploaded.name
    elif demo_choice != "None":
        ensure_demo_images()
        image = Image.open(SAMPLE_DIR / demo_choice).convert("RGB")
        filename = demo_choice

    if image is None:
        st.info("Upload an image or choose a demo image to start recognition.")
        return

    features, mask = extract_features(image, sensitivity=sensitivity)
    lib_pred = classify_library(features, filename=filename)
    ref_pred = classify_reference(features, reference_df) if reference_df is not None and not reference_df.empty else pd.DataFrame()
    pred = ref_pred if not ref_pred.empty else lib_pred

    top = pred.iloc[0].to_dict()
    category = str(top.get("Category", "Unknown")) if "Category" in pred.columns else "Custom reference"
    grade, action = ecological_grade(str(top.get("Predicted taxon", "Unknown")), category)

    st.divider()
    st.subheader("Recognition result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted taxon", str(top.get("Predicted taxon", "Unknown")))
    m2.metric("Confidence", f"{float(top.get('Confidence', 0)):.1%}")
    m3.metric("Detected count", int(features.get("count", 0)))
    m4.metric("Bioindicator", grade.split("/")[0].strip())

    st.markdown(f"<div class='success-card'><b>Management interpretation:</b> {action}</div>", unsafe_allow_html=True)

    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.image(image, caption="Original image", use_container_width=True)
    with img_col2:
        st.image(mask_to_image(mask), caption="Foreground mask", use_container_width=True)
    with img_col3:
        st.image(overlay_mask(image, mask), caption="Mask overlay", use_container_width=True)

    table_col, chart_col = st.columns([1.25, 1])
    with table_col:
        st.subheader("Top candidates")
        display_cols = [c for c in ["Predicted taxon", "Common name", "Category", "Confidence", "Ecological interpretation"] if c in pred.columns]
        st.dataframe(pred.head(5)[display_cols], use_container_width=True, hide_index=True)
    with chart_col:
        st.subheader("Confidence ranking")
        chart_df = pred.head(5).set_index("Predicted taxon")[["Confidence"]]
        st.bar_chart(chart_df, use_container_width=True)

    st.subheader("Extracted image features")
    feature_df = pd.DataFrame([{k: round(v, 4) for k, v in features.items()}])
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    record = make_result_record(
        filename=filename,
        features=features,
        prediction_df=pred,
        site=site,
        sample_date=str(sample_date),
        latitude=lat,
        longitude=lon,
        water_note=habitat,
    )
    record_df = pd.DataFrame([record])
    down_col1, down_col2 = st.columns(2)
    with down_col1:
        dataframe_download(record_df, "ai_river_biology_single_result.csv", "Download single-image CSV")
    with down_col2:
        st.download_button(
            "Download text report",
            data=report_text(record, pred).encode("utf-8"),
            file_name="ai_river_biology_report.txt",
            mime="text/plain",
            use_container_width=True,
        )


def page_batch(sensitivity: float, reference_df: Optional[pd.DataFrame]) -> None:
    st.header("Batch recognition")
    st.caption("Upload multiple images. The app will return one recognition record per file and export a CSV table.")

    meta_cols = st.columns(4)
    with meta_cols[0]:
        site = st.text_input("Shared site name", value="Batch survey reach")
    with meta_cols[1]:
        sample_date = st.date_input("Shared date", value=date.today())
    with meta_cols[2]:
        lat = st.number_input("Shared latitude", value=39.9042, format="%.6f")
    with meta_cols[3]:
        lon = st.number_input("Shared longitude", value=116.4074, format="%.6f")
    water_note = st.text_input("Shared field note", value="Batch upload from river biological survey.")

    uploads = st.file_uploader(
        "Upload batch images",
        type=["png", "jpg", "jpeg", "tif", "tiff", "webp"],
        accept_multiple_files=True,
    )
    use_demo = st.checkbox("Use three built-in demo images", value=False)

    files = []
    if uploads:
        files.extend(uploads)
    if use_demo:
        ensure_demo_images()
        for name in ["demo_fish.png", "demo_mayfly.png", "demo_mussel.png"]:
            files.append(open(SAMPLE_DIR / name, "rb"))

    if not files:
        st.info("Upload images or select the demo batch option.")
        return

    records = []
    previews = []
    progress = st.progress(0)
    for i, f in enumerate(files):
        try:
            image = load_image(f)
            filename = getattr(f, "name", f"image_{i + 1}.png")
            features, mask = extract_features(image, sensitivity=sensitivity)
            lib_pred = classify_library(features, filename=filename)
            ref_pred = classify_reference(features, reference_df) if reference_df is not None and not reference_df.empty else pd.DataFrame()
            pred = ref_pred if not ref_pred.empty else lib_pred
            records.append(make_result_record(filename, features, pred, site, str(sample_date), lat, lon, water_note))
            if len(previews) < 6:
                previews.append((filename, image, pred.iloc[0].to_dict()))
        except Exception as exc:  # Keep batch robust for mixed file sets.
            records.append({"file": getattr(f, "name", f"image_{i + 1}"), "error": str(exc)})
        progress.progress((i + 1) / len(files))

    df = pd.DataFrame(records)
    st.subheader("Batch results")
    st.dataframe(df, use_container_width=True, hide_index=True)
    dataframe_download(df, "ai_river_biology_batch_results.csv", "Download batch CSV")

    st.subheader("Preview gallery")
    cols = st.columns(3)
    for idx, (filename, image, top) in enumerate(previews):
        with cols[idx % 3]:
            st.image(image, caption=f"{filename}\n→ {top.get('Predicted taxon', 'Unknown')}", use_container_width=True)

    if "predicted_taxon" in df.columns:
        st.subheader("Community composition summary")
        comp = df["predicted_taxon"].value_counts().reset_index()
        comp.columns = ["Predicted taxon", "Image count"]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(comp, use_container_width=True, hide_index=True)
        with c2:
            st.bar_chart(comp.set_index("Predicted taxon"), use_container_width=True)


def page_reference(sensitivity: float) -> Optional[pd.DataFrame]:
    st.header("Custom reference-library training")
    st.caption("Upload a ZIP file with labelled folders, then the app uses k-nearest-neighbour matching from your own field images.")

    st.markdown(
        """
        **ZIP format example**

        ```text
        reference_images.zip
        ├── Cyprinidae fish/
        │   ├── fish_01.jpg
        │   └── fish_02.jpg
        ├── Ephemeroptera nymph/
        │   ├── mayfly_01.jpg
        │   └── mayfly_02.jpg
        └── Limnoperna fortunei/
            ├── mussel_01.jpg
            └── mussel_02.jpg
        ```
        """
    )

    zip_upload = st.file_uploader("Upload labelled reference ZIP", type=["zip"])
    if zip_upload is not None:
        with st.spinner("Building reference feature database..."):
            ref_df = build_reference_database(zip_upload, sensitivity=sensitivity)
        if ref_df.empty:
            st.error("No image files were found in the ZIP. Please check the folder structure.")
        else:
            st.session_state["reference_df"] = ref_df
            st.success(f"Reference library built: {len(ref_df)} images, {ref_df['label'].nunique()} labels.")

    ref_df = st.session_state.get("reference_df")
    if ref_df is not None and not ref_df.empty:
        st.subheader("Current reference database")
        st.dataframe(ref_df, use_container_width=True, hide_index=True)
        dataframe_download(ref_df, "ai_river_biology_reference_features.csv", "Download reference feature table")
        if st.button("Clear reference library"):
            st.session_state.pop("reference_df", None)
            st.rerun()
    else:
        st.info("No custom reference library is loaded. The app will use the built-in transparent prototype classifier.")
    return st.session_state.get("reference_df")


def page_library() -> None:
    st.header("Built-in taxon and indicator library")
    st.caption("This library supports the transparent prototype classifier and ecological interpretation layer.")
    st.dataframe(LIB_DF, use_container_width=True, hide_index=True)
    dataframe_download(LIB_DF, "ai_river_biology_taxon_library.csv", "Download taxon library CSV")

    st.subheader("How to interpret groups")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><b>Fish</b><br><span class='small-note'>Useful for habitat connectivity, refuge availability and river reach structure.</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><b>EPT insects</b><br><span class='small-note'>Ephemeroptera, Plecoptera and Trichoptera are commonly used as sensitive biological indicators.</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><b>Tolerant benthos</b><br><span class='small-note'>Chironomidae and Oligochaeta dominance can indicate fine sediment or organic enrichment.</span></div>", unsafe_allow_html=True)


def page_method() -> None:
    st.header("Method, limitations and future improvement")
    st.markdown(
        """
        ### Current prototype method
        1. **Image preprocessing**: resize and normalize the uploaded bioimage.
        2. **Foreground segmentation**: combine brightness thresholding, saturation cues and morphological cleanup to separate organisms from background.
        3. **Feature extraction**: calculate area ratio, aspect ratio, extent, compactness, edge density and RGB statistics.
        4. **Recognition**: compare extracted features with a built-in taxon prototype library or a user-uploaded labelled reference image library.
        5. **Ecological output**: convert the recognition result into river-management interpretation.

        ### What is already complete
        - Runnable website interface;
        - Single-image recognition;
        - Batch recognition;
        - Custom reference-library training from a labelled ZIP;
        - Count estimation, metadata recording, CSV export and text report.

        ### Limitations
        - The built-in classifier is a transparent course prototype, not a fully trained deep-learning model.
        - Real taxa with similar morphology may need expert verification.
        - Field images with complex background, overlapping organisms or poor focus will reduce accuracy.

        ### Future deep-learning version
        The next version can replace the transparent classifier with a YOLO / Mask R-CNN / Vision Transformer model trained on labelled fish and benthic macroinvertebrate images. The current website structure can remain unchanged: only the recognition backend needs to be upgraded.
        """
    )

    st.subheader("Pseudo-code of the recognition backend")
    st.code(
        """
        image = upload_bioimage()
        mask = segment_foreground(image)
        features = extract_shape_color_texture(mask, image)
        if custom_reference_library_exists:
            result = knn_match(features, reference_features)
        else:
            result = prototype_match(features, built_in_taxon_library)
        report = generate_taxon_count_location_interpretation(result)
        export(report, format="CSV/TXT")
        """,
        language="python",
    )


# ---------------------------
# Main app
# ---------------------------
def main() -> None:
    ensure_demo_images()
    inject_css()

    with st.sidebar:
        st.title("AI River Biology")
        page = st.radio(
            "Navigation",
            [
                "Home",
                "Single-image recognition",
                "Batch recognition",
                "Custom reference library",
                "Taxon library",
                "Method & limitations",
            ],
        )
        st.divider()
        sensitivity = st.slider("Segmentation sensitivity", 0.10, 0.95, 0.55, 0.05)
        st.caption("Higher sensitivity detects more foreground pixels; lower sensitivity is more conservative.")
        ref_df = st.session_state.get("reference_df")
        if ref_df is not None and not ref_df.empty:
            st.success(f"Custom reference active: {len(ref_df)} images / {ref_df['label'].nunique()} labels")
        else:
            st.info("Using built-in prototype classifier")

    if page == "Home":
        page_home()
    elif page == "Single-image recognition":
        page_single(sensitivity, st.session_state.get("reference_df"))
    elif page == "Batch recognition":
        page_batch(sensitivity, st.session_state.get("reference_df"))
    elif page == "Custom reference library":
        page_reference(sensitivity)
    elif page == "Taxon library":
        page_library()
    elif page == "Method & limitations":
        page_method()


if __name__ == "__main__":
    main()
