## Quora Duplicate Question Pairs - Clean Project

### Overview
This project trains a model to detect whether two questions are duplicates. It reproduces a precision of **0.847864162693393** on a held-out validation split and provides a Streamlit app for interactive predictions.

### Project Structure
- `train_model.py` — Training pipeline: preprocessing, feature building (TF‑IDF, TF‑IDF–weighted GloVe embeddings, numeric), RandomForest training, metric printout, and artifact saving.
- `app.py` — Streamlit app that loads saved artifacts and predicts duplicate vs not duplicate.
- `models/` — Saved artifacts (`model.joblib`, `tfidf.joblib`, `feature_meta.json`) created by `train_model.py`.
- `requirements.txt` — Dependencies for training and the app.
- `.gitignore` — Ignores large files and artifacts.

Optional: keep notebooks under `notebooks/` (not included by default).

### Setup
```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell
pip install -r requirements.txt
```

### Train and Reproduce Precision
```bash
python train_model.py
```
This prints the precision (expected: `0.847864162693393`) and saves:
- `models/model.joblib`
- `models/tfidf.joblib`
- `models/feature_meta.json`

Note: The first run downloads `glove-wiki-gigaword-100` (~128MB) and caches it.

### Run the App Locally
```bash
streamlit run app.py
```
Open the provided URL, enter two questions, and view the prediction and confidence.

### Deploy on Streamlit Cloud (recommended)
1. Push this repo to GitHub (exclude raw data files).
2. Create an app on Streamlit Community Cloud pointing to `app.py`.
3. The app installs `requirements.txt`; first startup downloads GloVe and caches it.
4. Optionally commit the `models/` folder after local training to skip any training online.

### Notes
- Keep large data files (`train.csv`, `test.csv`, `.zip`) out of version control; `.gitignore` already excludes them.
- If precision deviates, ensure you are using Python 3.10+ and the listed package versions.


