"""
可选脚本：从已命名浮游植物 PDF 中提取内置训练库。

用法：
    pip install pymupdf pillow pandas numpy
    python scripts/build_reference_from_pdf.py /path/to/已命名.pdf

注意：App 部署运行不需要该脚本，也不需要 pymupdf。
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

SPECIAL_SPLIT = {"四足十字藻四足十字藻": ["四足十字藻", "四足十字藻"]}


def parse_labels(page_text: str):
    phylum = None
    labels = []
    for line in page_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.fullmatch(r".{1,10}门", s):
            phylum = s
            continue
        parts = [x.strip() for x in re.split(r"\s{2,}", s) if x.strip()]
        for x in parts:
            labels.extend(SPECIAL_SPLIT.get(x, [x]))
    return phylum, labels


def main(pdf_path: str):
    base = Path(__file__).resolve().parents[1]
    img_dir = base / "reference_images"
    img_dir.mkdir(exist_ok=True)
    meta_path = base / "reference_metadata.csv"
    summary_path = base / "label_summary.csv"
    log_path = base / "dataset_build_log.json"

    doc = fitz.open(pdf_path)
    current_phylum = None
    rows = []
    skipped = []
    for pi, page in enumerate(doc, start=1):
        page_phylum, labels = parse_labels(page.get_text())
        if page_phylum:
            current_phylum = page_phylum
        imgs = page.get_images(full=True)
        m = min(len(labels), len(imgs))
        if len(labels) != len(imgs):
            skipped.append({"page": pi, "labels": len(labels), "images": len(imgs), "used": m})
        for ii, (imginfo, label) in enumerate(zip(imgs[:m], labels[:m]), start=1):
            xref = imginfo[0]
            base_img = doc.extract_image(xref)
            im = Image.open(io.BytesIO(base_img["image"])).convert("RGB")
            im.thumbnail((640, 640), Image.Resampling.LANCZOS)
            filename = f"p{pi:03d}_i{ii:02d}.jpg"
            im.save(img_dir / filename, quality=92, optimize=True)
            rows.append({
                "id": f"p{pi:03d}_i{ii:02d}",
                "page": pi,
                "image_index": ii,
                "phylum": current_phylum or "未标注门类",
                "taxon": label,
                "filename": filename,
            })

    with meta_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "page", "image_index", "phylum", "taxon", "filename"])
        w.writeheader(); w.writerows(rows)

    ph = defaultdict(Counter)
    for r in rows:
        ph[r["phylum"]][r["taxon"]] += 1
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["phylum", "taxon", "n_images"])
        for phy, c in ph.items():
            for taxon, n in c.most_common():
                w.writerow([phy, taxon, n])

    log_path.write_text(json.dumps({"n_images": len(rows), "skipped_mismatch": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"extracted {len(rows)} labelled images")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python build_reference_from_pdf.py /path/to/已命名.pdf")
    main(sys.argv[1])
