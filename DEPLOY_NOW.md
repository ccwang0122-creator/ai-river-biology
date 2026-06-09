# Deploy AI River Biology as a public web app

This folder is ready for Streamlit Community Cloud. After deployment, users can open a public link and start recognizing fish and benthic macroinvertebrates directly in the browser.

## Fastest method: Streamlit Community Cloud

1. Create a GitHub repository, for example: `ai-river-biology`.
2. Upload **all files in this folder** to the repository root. Make sure `app.py` is in the root, not inside another nested folder.
3. Open Streamlit Community Cloud.
4. Click **Create app** / **New app**.
5. Select your GitHub repository.
6. Set **Main file path** to:

```text
app.py
```

7. Click **Deploy**.
8. When deployment is finished, Streamlit will generate a public URL. Share that URL with your teacher/classmates.

## Alternative method: Hugging Face Spaces

1. Create a new Space.
2. Choose **SDK: Streamlit**.
3. Upload all files in this folder to the Space.
4. The Space will build automatically and provide a public web link.

## Local test before deployment

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Suggested public app title

AI River Biology: Automatic Fish and Benthic Macroinvertebrate Recognition

## Notes for course submission

- The app provides single-image recognition, batch recognition, custom reference-library recognition, taxon library, and downloadable results.
- The built-in classifier is a transparent prototype based on segmentation, morphology, color and texture features.
- For a formal monitoring system, the model should be improved using labelled biological images and deep-learning models such as YOLO, Mask R-CNN or Vision Transformer.
