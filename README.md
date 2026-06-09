# AI River Biology

**Automatic recognition of fish and benthic macroinvertebrates from river bioimages**  
Course prototype for *River Dynamics and Integrated River Management*.

## 1. App topic

**AI River Biology: Fish and Benthic Macroinvertebrate Recognition Website**

The website converts biological images collected during river surveys into:

- predicted taxon / species group;
- fish or benthic macroinvertebrate category;
- individual count estimate;
- sampling site, date and location record;
- ecological interpretation for river management;
- downloadable CSV and TXT reports.

## 2. Main functions

1. **Single-image recognition**  
   Upload one image and obtain segmentation, top taxon candidates, confidence, count and management interpretation.

2. **Batch recognition**  
   Upload multiple images and export a complete CSV result table.

3. **Custom reference-library training**  
   Upload a ZIP file with labelled folders. The app will build a feature database and classify new images using k-nearest-neighbour matching.

4. **Built-in taxon library**  
   Includes common river fish and benthic macroinvertebrate groups such as Cyprinidae fish, loach-type fish, Ephemeroptera, Plecoptera, Trichoptera, Chironomidae, Oligochaeta, Gastropoda, Bivalvia and *Limnoperna fortunei*.

5. **Downloadable outputs**  
   CSV, reference feature table and text recognition report.

## 3. How to run locally

```bash
cd ai_river_biology_app
pip install -r requirements.txt
streamlit run app.py
```

The website will open in your browser, usually at:

```text
http://localhost:8501
```

## 4. How to deploy on Streamlit Cloud

1. Create a GitHub repository.
2. Upload all files in this folder.
3. Open Streamlit Cloud and create a new app.
4. Select `app.py` as the main file.
5. Click **Deploy**.

## 5. Reference ZIP format

Use labelled folders. Folder names become taxon labels.

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

## 6. Method summary

The prototype uses a transparent pipeline:

```text
Bioimage upload
→ preprocessing
→ foreground segmentation
→ shape/color/texture feature extraction
→ prototype matching or custom reference-library matching
→ taxon, count, location and ecological interpretation
→ CSV/TXT download
```

The built-in classifier is intentionally lightweight for course demonstration. For formal biological monitoring, results should be verified by taxonomic experts and improved using a labelled training dataset and deep-learning models such as YOLO, Mask R-CNN or Vision Transformer.

## 7. Suggested presentation script

This app belongs to the AI River Biology module. Its input is river biological imagery, including fish photos and benthic macroinvertebrate photos. The app first segments the target organism, extracts transparent image features, then performs recognition using either a built-in taxon prototype library or a custom reference image library. The output includes predicted taxon, confidence, estimated count, sampling location and ecological interpretation. In this way, the app links biological image recognition with river habitat assessment and river management.
