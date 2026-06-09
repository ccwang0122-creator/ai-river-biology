"""
AI River Biology — real-photo reference version
Automatic recognition of freshwater fish families and benthic macroinvertebrate groups.

This course app uses real reference photographs retrieved from Wikimedia Commons categories
instead of cartoon demo shapes. It performs lightweight image segmentation, feature extraction,
and nearest-reference matching. It is intended as an explainable prototype for teaching/demo use.
"""

from __future__ import annotations

import io
import math
import zipfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage

APP_DIR = Path(__file__).resolve().parent
SOURCE_CSV = APP_DIR / "reference_sources.csv"

st.set_page_config(
    page_title="AI River Biology",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_AGENT = "AI-River-Biology-course-prototype/2.0 (Wikimedia Commons API; Streamlit app)"

REFERENCE_TAXA: List[Dict[str, str]] = [
    {"taxon": "Cyprinidae fish", "common": "Carp / minnow type fish", "category": "Fish", "commons": "Cyprinidae", "indicator": "Common river fish family; useful for connectivity and habitat continuity screening.", "management": "Check whether the sample point has connected flow paths, refuge habitat and fish passage conditions."},
    {"taxon": "Salmonidae fish", "common": "Salmon / trout type fish", "category": "Fish", "commons": "Salmonidae", "indicator": "Cold-water and oxygen-demanding fish group in many rivers.", "management": "Interpret together with temperature, dissolved oxygen and longitudinal connectivity."},
    {"taxon": "Siluridae fish", "common": "Catfish type fish", "category": "Fish", "commons": "Siluridae", "indicator": "Large-bodied predatory/benthic fish group; often linked with deeper pools and complex habitats.", "management": "Use as a clue for pool habitats and food-web structure, not as a direct water-quality indicator."},
    {"taxon": "Cobitidae fish", "common": "Loach type fish", "category": "Fish", "commons": "Cobitidae", "indicator": "Small benthic fish often associated with shallow margins and sandy/muddy substrates.", "management": "Useful for habitat heterogeneity and substrate-condition interpretation."},
    {"taxon": "Percidae fish", "common": "Perch / darter type fish", "category": "Fish", "commons": "Percidae", "indicator": "Perch/darter group; some taxa are habitat specialists in flowing waters.", "management": "Interpret together with substrate, channel complexity and flow condition."},
    {"taxon": "Centrarchidae fish", "common": "Sunfish / bass type fish", "category": "Fish", "commons": "Centrarchidae", "indicator": "Still-water or slow-flow fish group in many systems.", "management": "Can indicate pool, backwater or low-velocity habitat availability."},
    {"taxon": "Gobiidae fish", "common": "Goby type fish", "category": "Fish", "commons": "Gobiidae", "indicator": "Small benthic fish group; some species are associated with substrate and near-bed habitat.", "management": "Use for benthic habitat screening; check local species list for invasive/native status."},
    {"taxon": "Ephemeroptera larva/nymph", "common": "Mayfly nymph", "category": "Benthic macroinvertebrate", "commons": "Ephemeroptera larvae", "indicator": "Generally sensitive EPT group, often used in biological water-quality assessment.", "management": "Higher occurrence can support better habitat and oxygen-condition interpretation."},
    {"taxon": "Plecoptera larva/nymph", "common": "Stonefly nymph", "category": "Benthic macroinvertebrate", "commons": "Plecoptera larvae", "indicator": "Sensitive EPT group, often linked with cold, oxygen-rich, coarse-substrate streams.", "management": "Occurrence is useful evidence for high-quality benthic habitat."},
    {"taxon": "Trichoptera larva", "common": "Caddisfly larva", "category": "Benthic macroinvertebrate", "commons": "Trichoptera larvae and hides", "indicator": "EPT group; many taxa reflect substrate stability and organic-matter processing.", "management": "Interpret with substrate, detritus, woody debris and flow microhabitat."},
    {"taxon": "Chironomidae larva", "common": "Non-biting midge larva", "category": "Benthic macroinvertebrate", "commons": "Chironomidae larvae", "indicator": "Tolerant group; dominance may suggest organic enrichment or fine sediment deposition.", "management": "Check with nutrients, sediment and dissolved oxygen before making management conclusions."},
    {"taxon": "Oligochaeta", "common": "Aquatic worm", "category": "Benthic macroinvertebrate", "commons": "Oligochaeta", "indicator": "Tolerant worm group often associated with fine sediment and organic matter.", "management": "Useful for screening sedimentation or organic pollution pressure."},
    {"taxon": "Gastropoda", "common": "Freshwater / river snail", "category": "Benthic macroinvertebrate", "commons": "Gastropoda", "indicator": "Grazing mollusc group; response depends on periphyton, substrate and water quality.", "management": "Interpret together with macrophytes, periphyton and hard-substrate availability."},
    {"taxon": "Bivalvia", "common": "Freshwater mussel / clam", "category": "Benthic macroinvertebrate", "commons": "Bivalvia", "indicator": "Filter-feeding mollusc group related to bed stability, hydrological connectivity and water quality.", "management": "Screen for stable substrate, sediment pressure and ecological filtration function."},
    {"taxon": "Limnoperna fortunei", "common": "Golden mussel", "category": "Benthic macroinvertebrate", "commons": "Limnoperna fortunei", "indicator": "Invasive freshwater mytilid; important for biofouling and water-transfer infrastructure risk.", "management": "Recommend rapid verification, density survey and infrastructure risk screening."},
]

# Fallback prototypes are only used if online reference images cannot be fetched.
FALLBACK_PROTOTYPES = {
    "Cyprinidae fish": [0.20, 3.8, 0.55, 0.20, 0.10, 0.43, 0.49, 0.42, 0.15],
    "Salmonidae fish": [0.20, 4.2, 0.56, 0.20, 0.11, 0.45, 0.50, 0.45, 0.16],
    "Siluridae fish": [0.22, 3.2, 0.50, 0.18, 0.13, 0.38, 0.39, 0.35, 0.17],
    "Cobitidae fish": [0.12, 5.5, 0.45, 0.13, 0.10, 0.43, 0.42, 0.34, 0.18],
    "Percidae fish": [0.16, 3.3, 0.50, 0.19, 0.14, 0.45, 0.47, 0.38, 0.18],
    "Centrarchidae fish": [0.20, 2.2, 0.55, 0.28, 0.12, 0.48, 0.50, 0.38, 0.18],
    "Gobiidae fish": [0.12, 2.7, 0.45, 0.22, 0.14, 0.42, 0.40, 0.34, 0.18],
    "Ephemeroptera larva/nymph": [0.08, 2.8, 0.36, 0.15, 0.28, 0.48, 0.44, 0.34, 0.22],
    "Plecoptera larva/nymph": [0.07, 2.9, 0.34, 0.14, 0.32, 0.36, 0.34, 0.29, 0.24],
    "Trichoptera larva": [0.10, 2.0, 0.40, 0.24, 0.24, 0.52, 0.49, 0.38, 0.21],
    "Chironomidae larva": [0.05, 6.5, 0.25, 0.10, 0.22, 0.58, 0.32, 0.28, 0.20],
    "Oligochaeta": [0.04, 8.0, 0.18, 0.08, 0.18, 0.55, 0.38, 0.34, 0.16],
    "Gastropoda": [0.10, 1.3, 0.58, 0.50, 0.16, 0.45, 0.40, 0.32, 0.18],
    "Bivalvia": [0.13, 1.8, 0.62, 0.55, 0.12, 0.43, 0.39, 0.32, 0.15],
    "Limnoperna fortunei": [0.09, 1.9, 0.57, 0.48, 0.15, 0.52, 0.46, 0.28, 0.18],
}

FEATURE_NAMES = ["area_ratio", "aspect_ratio", "extent", "compactness", "edge_density", "mean_r", "mean_g", "mean_b", "gray_std"]
FEATURE_SCALE = np.array([0.18, 4.0, 0.38, 0.32, 0.22, 0.22, 0.22, 0.22, 0.15], dtype=float)


def css() -> None:
    st.markdown("""
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
    .hero {padding:1.3rem 1.5rem;border-radius:20px;background:linear-gradient(135deg,#EAF7FF,#F7FBF8 55%,#FFF6E8);border:1px solid #D7E7F5;margin-bottom:1rem;}
    .hero h1 {margin:0 0 .4rem 0;color:#103A5C;font-size:2.15rem;}
    .hero p {margin:0;color:#475569;font-size:1.02rem;}
    .pill {display:inline-block;padding:.25rem .65rem;margin:.15rem .2rem .15rem 0;border-radius:999px;background:#EEF6FF;border:1px solid #C8E4FF;color:#174E79;font-size:.82rem;}
    .card {border:1px solid #E2E8F0;border-radius:18px;padding:1rem;background:#fff;box-shadow:0 4px 12px rgba(15,23,42,.04);min-height:110px;}
    .note {border:1px solid #FBD38D;border-radius:14px;padding:.85rem;background:#FFFBEB;color:#7C4A03;}
    .ok {border:1px solid #BBF7D0;border-radius:14px;padding:.85rem;background:#F0FDF4;color:#14532D;}
    .small {font-size:.86rem;color:#64748B;}
    div[data-testid="stMetric"] {background:#fff;border:1px solid #E2E8F0;padding:.8rem;border-radius:16px;}
    </style>
    """, unsafe_allow_html=True)


def hero() -> None:
    st.markdown("""
    <div class="hero">
      <h1>🐟 AI River Biology</h1>
      <p><b>Real-photo reference version:</b> automatic recognition of freshwater fish families and benthic macroinvertebrate groups from river bioimages.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="pill">Real reference photos</span><span class="pill">Wikimedia Commons</span><span class="pill">Fish families</span><span class="pill">Benthic macroinvertebrates</span><span class="pill">CSV report</span>', unsafe_allow_html=True)


def pil_open_from_upload(uploaded) -> Image.Image:
    return Image.open(uploaded).convert("RGB")


def resize_for_processing(img: Image.Image, max_side: int = 720) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def segment_foreground(img: Image.Image, sensitivity: float = 0.45) -> np.ndarray:
    """Foreground segmentation based on border-background color distance plus morphology."""
    img = resize_for_processing(img)
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w, _ = arr.shape
    border = max(6, min(h, w) // 30)
    border_pixels = np.concatenate([
        arr[:border, :, :].reshape(-1, 3), arr[-border:, :, :].reshape(-1, 3),
        arr[:, :border, :].reshape(-1, 3), arr[:, -border:, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border_pixels, axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    # sensitivity higher = easier to include foreground
    threshold = max(0.035, 0.20 - 0.16 * sensitivity)
    mask = dist > threshold

    # If the background is not uniform, use grayscale contrast to repair the mask.
    gray = np.mean(arr, axis=2)
    q_low, q_high = np.quantile(gray, [0.05, 0.95])
    contrast_mask = (gray < q_low + (q_high - q_low) * 0.38) | (gray > q_high - (q_high - q_low) * 0.38)
    if mask.mean() < 0.015:
        mask = contrast_mask
    if mask.mean() > 0.75:
        # too much selected: keep the strongest color-distance pixels only
        mask = dist > np.quantile(dist, 0.65)

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)

    labels, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    min_size = max(50, int(0.002 * h * w))
    keep_ids = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    if not keep_ids:
        keep_ids = [int(np.argmax(sizes) + 1)]
    clean = np.isin(labels, keep_ids)
    return clean.astype(bool)


def feature_vector(img: Image.Image, sensitivity: float = 0.45) -> Tuple[np.ndarray, np.ndarray, int]:
    img = resize_for_processing(img)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mask = segment_foreground(img, sensitivity)
    h, w = mask.shape
    area = float(mask.sum())
    area_ratio = area / max(1.0, h * w)

    labels, n = ndimage.label(mask)
    count = 0
    if n > 0:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        count = int(np.sum(np.array(sizes) > max(40, 0.001 * h * w)))
    count = max(1, count)

    if area <= 0:
        return np.zeros(len(FEATURE_NAMES)), mask, count

    ys, xs = np.where(mask)
    box_h = max(1, int(ys.max() - ys.min() + 1))
    box_w = max(1, int(xs.max() - xs.min() + 1))
    aspect = max(box_w / box_h, box_h / box_w)
    extent = area / (box_w * box_h)

    eroded = ndimage.binary_erosion(mask)
    perimeter = float(np.logical_xor(mask, eroded).sum())
    compactness = float(4 * math.pi * area / (perimeter * perimeter + 1e-6))
    edge_density = perimeter / (area + 1e-6)
    mean_rgb = arr[mask].mean(axis=0)
    gray = arr.mean(axis=2)
    gray_std = float(gray[mask].std())
    feat = np.array([area_ratio, aspect, extent, compactness, edge_density, mean_rgb[0], mean_rgb[1], mean_rgb[2], gray_std], dtype=float)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat, mask, count


def mask_overlay(img: Image.Image, mask: np.ndarray) -> Image.Image:
    img = resize_for_processing(img)
    arr = np.asarray(img).copy()
    overlay = arr.copy()
    overlay[mask] = (0.55 * overlay[mask] + np.array([255, 60, 45]) * 0.45).astype(np.uint8)
    return Image.fromarray(overlay)


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_commons_records(category_name: str, limit: int = 6) -> List[Dict[str, str]]:
    """Return thumbnail URLs and metadata from a Wikimedia Commons category."""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category_name}",
        "gcmtype": "file",
        "gcmlimit": max(1, min(30, limit * 3)),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": 460,
        "format": "json",
        "origin": "*",
    }
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    records = []
    for page in pages.values():
        ii = page.get("imageinfo", [{}])[0]
        mime = ii.get("mime", "")
        if not mime.startswith("image/"):
            continue
        # Skip GIF/SVG because PIL handling is inconsistent in cloud demo.
        if mime in {"image/svg+xml", "image/gif", "image/tiff"}:
            continue
        meta = ii.get("extmetadata", {}) or {}
        def meta_value(k: str) -> str:
            return str(meta.get(k, {}).get("value", ""))
        records.append({
            "file_title": page.get("title", ""),
            "thumb_url": ii.get("thumburl") or ii.get("url"),
            "original_url": ii.get("descriptionurl") or ii.get("url"),
            "artist": meta_value("Artist"),
            "license": meta_value("LicenseShortName"),
            "credit": meta_value("Credit"),
        })
        if len(records) >= limit:
            break
    return records


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def download_image_bytes(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    return r.content


@st.cache_data(show_spinner=True, ttl=24 * 3600)
def build_real_reference_library(limit_per_taxon: int = 5, sensitivity: float = 0.45) -> pd.DataFrame:
    rows = []
    for tax in REFERENCE_TAXA:
        try:
            records = fetch_commons_records(tax["commons"], limit=limit_per_taxon)
        except Exception:
            records = []
        for rec in records:
            try:
                content = download_image_bytes(rec["thumb_url"])
                img = Image.open(io.BytesIO(content)).convert("RGB")
                # Improve low-contrast museum shell/specimen photos slightly.
                img = ImageEnhance.Contrast(img).enhance(1.05)
                feat, _, _ = feature_vector(img, sensitivity=sensitivity)
                rows.append({
                    "taxon": tax["taxon"], "common": tax["common"], "category": tax["category"],
                    "commons_category": tax["commons"], "indicator": tax["indicator"], "management": tax["management"],
                    "file_title": rec["file_title"], "thumb_url": rec["thumb_url"], "source": rec["original_url"],
                    "license": rec["license"], "artist": rec["artist"], "credit": rec["credit"],
                    **{name: float(val) for name, val in zip(FEATURE_NAMES, feat)}
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


def fallback_library() -> pd.DataFrame:
    rows = []
    meta = {t["taxon"]: t for t in REFERENCE_TAXA}
    for taxon, proto in FALLBACK_PROTOTYPES.items():
        tax = meta[taxon]
        rows.append({
            "taxon": taxon, "common": tax["common"], "category": tax["category"], "commons_category": tax["commons"],
            "indicator": tax["indicator"], "management": tax["management"], "file_title": "Fallback morphology prototype",
            "thumb_url": "", "source": "", "license": "", "artist": "", "credit": "",
            **{name: float(val) for name, val in zip(FEATURE_NAMES, proto)}
        })
    return pd.DataFrame(rows)


def classify(img: Image.Image, sensitivity: float, limit_per_taxon: int) -> Tuple[pd.DataFrame, np.ndarray, int, pd.DataFrame]:
    feat, mask, count = feature_vector(img, sensitivity=sensitivity)
    refs = build_real_reference_library(limit_per_taxon=limit_per_taxon, sensitivity=sensitivity)
    used_fallback = False
    if refs.empty or refs[FEATURE_NAMES].isna().all().all() or refs["taxon"].nunique() < 6:
        refs = fallback_library()
        used_fallback = True

    X = refs[FEATURE_NAMES].to_numpy(dtype=float)
    dist = np.linalg.norm((X - feat) / FEATURE_SCALE, axis=1)
    # Image-level similarity. Smaller distance gives higher score.
    refs = refs.copy()
    refs["distance"] = dist
    refs["image_score"] = np.exp(-dist / 2.6)

    # Group-level score: average top 3 reference-image scores for each taxon.
    out_rows = []
    for taxon, g in refs.groupby("taxon"):
        g2 = g.sort_values("image_score", ascending=False).head(3)
        score = float(g2["image_score"].mean())
        first = g2.iloc[0]
        out_rows.append({
            "Predicted taxon": taxon,
            "Common name": first["common"],
            "Category": first["category"],
            "Score": score,
            "Best reference": first["file_title"],
            "Reference source": first["source"],
            "License": first["license"],
            "Indicator meaning": first["indicator"],
            "Management interpretation": first["management"],
            "Reference mode": "fallback prototype" if used_fallback else "real Wikimedia Commons photos",
        })
    results = pd.DataFrame(out_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    total = results["Score"].sum()
    results["Confidence"] = results["Score"] / total if total > 0 else 0
    return results, mask, count, refs


def candidate_chart(df: pd.DataFrame) -> plt.Figure:
    top = df.head(6).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(top["Predicted taxon"], top["Confidence"] * 100)
    ax.set_xlabel("Relative confidence (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(35, float((top["Confidence"] * 100).max()) * 1.2 if len(top) else 100))
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    return fig


def render_result(img: Image.Image, results: pd.DataFrame, mask: np.ndarray, count: int, site: str, sample_date: date) -> pd.DataFrame:
    top = results.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted taxon", str(top["Predicted taxon"]))
    c2.metric("Confidence", f"{float(top['Confidence']) * 100:.1f}%")
    c3.metric("Detected count", count)
    c4.metric("Reference mode", str(top["Reference mode"]))
    st.markdown(f"<div class='ok'><b>Management interpretation:</b> {top['Management interpretation']}</div>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.image(resize_for_processing(img), caption="Original image", use_container_width=True)
    with p2:
        st.image(mask_image(mask), caption="Foreground mask", use_container_width=True)
    with p3:
        st.image(mask_overlay(img, mask), caption="Mask overlay", use_container_width=True)

    st.subheader("Top candidates")
    show = results[["Predicted taxon", "Common name", "Category", "Confidence", "Best reference", "License"]].head(6).copy()
    show["Confidence"] = (show["Confidence"] * 100).map(lambda x: f"{x:.1f}%")
    left, right = st.columns([1.15, .85])
    with left:
        st.dataframe(show, use_container_width=True, hide_index=True)
    with right:
        st.pyplot(candidate_chart(results), use_container_width=True)

    report = results.head(10).copy()
    report.insert(0, "Sample date", str(sample_date))
    report.insert(0, "Site / location", site)
    report.insert(0, "Detected count", count)
    csv = report.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download CSV report", csv, "ai_river_biology_result.csv", "text/csv")

    text = f"AI River Biology recognition report\nSite: {site}\nDate: {sample_date}\nPredicted taxon: {top['Predicted taxon']}\nConfidence: {float(top['Confidence'])*100:.1f}%\nDetected count: {count}\nReference mode: {top['Reference mode']}\nIndicator meaning: {top['Indicator meaning']}\nManagement interpretation: {top['Management interpretation']}\nBest reference: {top['Best reference']}\nSource: {top['Reference source']}\nLicense: {top['License']}\n"
    st.download_button("Download TXT summary", text.encode("utf-8"), "ai_river_biology_summary.txt", "text/plain")
    return report


def home_page() -> None:
    hero()
    st.markdown("### Workflow")
    cols = st.columns(5)
    steps = [
        ("1. Input images", "Fish or benthic macroinvertebrate field/scope photos."),
        ("2. Real reference library", "Fetch labelled photos from Wikimedia Commons taxon categories."),
        ("3. Segment", "Separate organisms from background and estimate count."),
        ("4. Match", "Compare extracted morphology/color features with real references."),
        ("5. Output", "Taxon, confidence, count, management note and CSV report."),
    ]
    for col, (title, body) in zip(cols, steps):
        col.markdown(f"<div class='card'><b>{title}</b><br><span class='small'>{body}</span></div>", unsafe_allow_html=True)
    st.markdown("### What changed in this version")
    st.markdown("""
    - The cartoon `demo_fish.png / demo_mayfly.png / demo_mussel.png` are no longer used as the main reference library.
    - The app now builds a **real-photo reference set** from Wikimedia Commons categories covering common freshwater fish families and benthic macroinvertebrate groups.
    - The recognition level is set to **family / major taxonomic group**, which is more realistic for a course prototype than claiming exact species-level AI identification from a tiny dataset.
    """)
    st.markdown("<div class='note'><b>Prototype note:</b> This is still not a production-grade taxonomic model. It is an explainable teaching app. For field use, you should add your own labelled photos from the field trip in the Custom reference library page.</div>", unsafe_allow_html=True)


def single_page() -> None:
    hero()
    st.subheader("Single-image recognition")
    left, right = st.columns([.85, 1.15])
    with left:
        up = st.file_uploader("Upload a fish or benthic macroinvertebrate photo", type=["png", "jpg", "jpeg", "webp"])
        site = st.text_input("Sampling site / location", "River field site A")
        sample_date = st.date_input("Sampling date", value=date.today())
        st.caption("Best test images: clear side-view fish photos, macroinvertebrates on a light tray, or microscope images with simple background.")
    if up is None:
        st.info("Upload one real photo to start recognition.")
        return
    img = pil_open_from_upload(up)
    with st.spinner("Building real reference library and matching image..."):
        results, mask, count, refs = classify(img, sensitivity=st.session_state.sensitivity, limit_per_taxon=st.session_state.limit_per_taxon)
    render_result(img, results, mask, count, site, sample_date)


def batch_page() -> None:
    hero()
    st.subheader("Batch recognition")
    files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    site = st.text_input("Sampling site / location", "Batch field survey")
    sample_date = st.date_input("Sampling date", value=date.today())
    if not files:
        st.info("Upload two or more images to generate a batch CSV.")
        return
    rows = []
    progress = st.progress(0)
    for i, f in enumerate(files):
        img = pil_open_from_upload(f)
        results, mask, count, refs = classify(img, sensitivity=st.session_state.sensitivity, limit_per_taxon=st.session_state.limit_per_taxon)
        top = results.iloc[0]
        rows.append({
            "file": f.name, "site": site, "date": str(sample_date), "predicted_taxon": top["Predicted taxon"],
            "common_name": top["Common name"], "category": top["Category"], "confidence_percent": round(float(top["Confidence"]) * 100, 2),
            "detected_count": count, "reference_mode": top["Reference mode"], "management_interpretation": top["Management interpretation"],
            "best_reference": top["Best reference"], "reference_source": top["Reference source"], "license": top["License"],
        })
        progress.progress((i + 1) / len(files))
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download batch CSV", df.to_csv(index=False).encode("utf-8-sig"), "ai_river_biology_batch_results.csv", "text/csv")


def reference_page() -> None:
    hero()
    st.subheader("Real reference library")
    st.write("The app retrieves labelled real photographs from Wikimedia Commons categories. Click below to preview what the deployed app is using.")
    if st.button("Build / refresh real reference preview"):
        st.cache_data.clear()
        st.rerun()
    with st.spinner("Loading real reference images..."):
        refs = build_real_reference_library(limit_per_taxon=st.session_state.limit_per_taxon, sensitivity=st.session_state.sensitivity)
    if refs.empty:
        st.error("The app could not reach Wikimedia Commons right now. It will use fallback morphology prototypes until the connection is available.")
    else:
        st.success(f"Loaded {len(refs)} real reference photos from {refs['taxon'].nunique()} taxonomic groups.")
        st.dataframe(refs[["taxon", "common", "category", "commons_category", "file_title", "license", "source"]].head(80), use_container_width=True, hide_index=True)
        st.markdown("#### Preview examples")
        for taxon, g in refs.groupby("taxon"):
            with st.expander(taxon, expanded=False):
                cols = st.columns(min(4, len(g)))
                for col, (_, row) in zip(cols, g.head(4).iterrows()):
                    col.image(row["thumb_url"], caption=f"{row['file_title']}\n{row['license']}", use_container_width=True)
    st.markdown("#### Source categories")
    src = pd.read_csv(SOURCE_CSV) if SOURCE_CSV.exists() else pd.DataFrame(REFERENCE_TAXA)
    st.dataframe(src, use_container_width=True, hide_index=True)

    st.markdown("### Add your own field-trip reference library")
    st.write("Upload a ZIP with subfolders named by taxa, for example `Cyprinidae fish/photo1.jpg`, `Ephemeroptera larva/photo2.jpg`. The current web demo mainly uses Wikimedia Commons; your own labelled photos are recommended for the final field-trip version.")
    custom_zip = st.file_uploader("Upload labelled reference ZIP", type=["zip"])
    if custom_zip is not None:
        st.info("ZIP received. For this classroom version, keep it as evidence in the presentation; full persistent retraining requires saving the files to a database or repository.")
        with zipfile.ZipFile(custom_zip) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        st.write(f"Detected {len(names)} image files in ZIP.")
        st.code("\n".join(names[:30]))


def taxon_page() -> None:
    hero()
    st.subheader("Taxon library")
    df = pd.DataFrame([{
        "Taxon": t["taxon"], "Common name": t["common"], "Category": t["category"],
        "Commons category": t["commons"], "Indicator meaning": t["indicator"], "Management interpretation": t["management"]
    } for t in REFERENCE_TAXA])
    st.dataframe(df, use_container_width=True, hide_index=True)


def method_page() -> None:
    hero()
    st.subheader("Method & limitations")
    st.markdown("""
    **Method.** The app queries Wikimedia Commons taxon categories, downloads thumbnail reference photos, segments organisms from the background, extracts transparent image features, and compares the uploaded image with the reference feature library. The output is a ranked taxonomic group, relative confidence, count estimate and ecological interpretation.

    **Why family/group level?** Exact species recognition requires a large curated training dataset and expert labels. For a course prototype, family or major-group recognition is more defensible and easier to explain.

    **Limitations.** Results depend strongly on image quality, background complexity, organism orientation and reference-library coverage. It should not be used as a formal taxonomic determination without manual verification.
    """)
    st.markdown("#### Extracted feature set")
    st.write(", ".join(FEATURE_NAMES))


def sidebar() -> str:
    st.sidebar.title("AI River Biology")
    page = st.sidebar.radio("Navigation", ["Home", "Single-image recognition", "Batch recognition", "Real reference library", "Taxon library", "Method & limitations"])
    st.sidebar.divider()
    st.session_state.sensitivity = st.sidebar.slider("Segmentation sensitivity", 0.10, 0.90, 0.42, 0.05)
    st.sidebar.caption("Higher sensitivity detects more foreground pixels; lower sensitivity is more conservative.")
    st.session_state.limit_per_taxon = st.sidebar.slider("Real reference photos per taxon", 2, 10, 5, 1)
    st.sidebar.success("Using real Wikimedia Commons reference photos")
    return page


def main() -> None:
    css()
    page = sidebar()
    if page == "Home":
        home_page()
    elif page == "Single-image recognition":
        single_page()
    elif page == "Batch recognition":
        batch_page()
    elif page == "Real reference library":
        reference_page()
    elif page == "Taxon library":
        taxon_page()
    else:
        method_page()


if __name__ == "__main__":
    main()
