# AI River Biology — Real-photo Reference Version

Automatic recognition of freshwater fish families and benthic macroinvertebrate groups from river biological images.

## App topic

**AI River Biology: Fish and Benthic Macroinvertebrate Recognition Website**

This version replaces the original cartoon demo images with a **real-photo reference library**. The app retrieves labelled reference photographs from Wikimedia Commons taxon categories and uses them for explainable image matching.

## Recognition scope

The app recognizes family-level or major-group categories, including:

- Cyprinidae, Salmonidae, Siluridae, Cobitidae, Percidae, Centrarchidae and Gobiidae fish;
- Ephemeroptera, Plecoptera, Trichoptera and Chironomidae larvae/nymphs;
- Oligochaeta, Gastropoda, Bivalvia and Limnoperna fortunei.

## Workflow

1. Upload a river bioimage.
2. Build a real-photo reference library from Wikimedia Commons categories.
3. Segment the organism from the image background.
4. Extract shape, color and texture features.
5. Match the uploaded image with reference photos.
6. Output predicted taxon, confidence, count estimate, ecological interpretation and CSV/TXT reports.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

- Repository: your GitHub repository
- Branch: `main`
- Main file path: `app.py`

## Important note

This is a course prototype, not a production-grade taxonomic identification system. Formal biological identification still requires expert verification. The best way to improve the model is to add labelled reference images from your own field trip.

## Image sources

Reference images are retrieved from Wikimedia Commons categories listed in `reference_sources.csv`. Each file keeps its source page and license metadata in the app output.
