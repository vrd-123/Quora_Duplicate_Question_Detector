import json
import gc
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix, hstack
import gensim.downloader as api
import joblib


@st.cache_resource
def load_nltk():
	nltk.download('punkt', quiet=True)
	nltk.download('wordnet', quiet=True)
	return True


@st.cache_resource
def load_artifacts():
	model = joblib.load('models/model.joblib')
	tfidf = joblib.load('models/tfidf.joblib')
	with open('models/feature_meta.json', 'r', encoding='utf-8') as f:
		meta = json.load(f)
	return model, tfidf, meta


@st.cache_resource
def load_embeddings():
	# ~128MB download on first run; cached afterwards
	return api.load('glove-wiki-gigaword-100')


def clean_text(text: str, lemm: WordNetLemmatizer) -> str:
	text = str(text).lower()
	# keep only words and whitespace
	import re
	text = re.sub(r'[^\w\s]', '', text)
	tokens = word_tokenize(text)
	return " ".join(lemm.lemmatize(t) for t in tokens)


def build_numeric_features(q1: str, q2: str) -> np.ndarray:
	q1_len = len(q1)
	q2_len = len(q2)
	q1_num_words = len(q1.split())
	q2_num_words = len(q2.split())
	s1 = set(q1.split())
	s2 = set(q2.split())
	words_common = len(s1 & s2)
	words_total = len(s1) + len(s2)
	word_share = (words_common / (words_total if words_total != 0 else 1))
	return np.array([q1_len, q2_len, q1_num_words, q2_num_words, words_common, words_total, word_share], dtype=np.float32)


def tfidf_weighted_avg(text: str, w2v_model, index_to_idf: dict, emb_dim: int) -> np.ndarray:
	words = text.split()
	vecs = []
	weights = []
	for w in words:
		if (w in w2v_model) and (w in index_to_idf):
			vecs.append(w2v_model[w])
			weights.append(index_to_idf[w])
	if not vecs:
		return np.zeros(emb_dim, dtype=np.float32)
	vecs_arr = np.asarray(vecs, dtype=np.float32)
	weights_arr = np.asarray(weights, dtype=np.float32).reshape((-1, 1))
	weighted = (vecs_arr * weights_arr).sum(axis=0) / (weights_arr.sum() + 1e-9)
	return weighted.astype(np.float32)


def main():
	st.set_page_config(page_title="Quora Duplicate Question Detector", page_icon="❓")
	st.title("Quora Duplicate Question Detector")
	st.write("Enter two questions to check if they are duplicates.")

	load_nltk()
	model, tfidf, meta = load_artifacts()
	w2v_model = load_embeddings()
	lemm = WordNetLemmatizer()

	# Build idf mapping for tokens in tfidf vocab
	vocab = tfidf.vocabulary_
	idf = tfidf.idf_
	index_to_idf = {token: float(idf[idx]) for token, idx in vocab.items()}
	emb_dim = int(meta.get('embedding_dim', 100))

	with st.form("dup_form"):
		q1_input = st.text_area("Question 1", height=100)
		q2_input = st.text_area("Question 2", height=100)
		submit = st.form_submit_button("Predict")

	if submit:
		if not q1_input.strip() or not q2_input.strip():
			st.warning("Please provide both questions.")
			return

		# Preprocess
		q1_clean = clean_text(q1_input, lemm)
		q2_clean = clean_text(q2_input, lemm)

		# Features
		num_features = build_numeric_features(q1_clean, q2_clean).reshape(1, -1)
		num_sparse = csr_matrix(num_features)
		q1_tfidf = tfidf.transform([q1_clean])
		q2_tfidf = tfidf.transform([q2_clean])

		q1_emb = tfidf_weighted_avg(q1_clean, w2v_model, index_to_idf, emb_dim).reshape(1, -1)
		q2_emb = tfidf_weighted_avg(q2_clean, w2v_model, index_to_idf, emb_dim).reshape(1, -1)
		emb = np.hstack([q1_emb, q2_emb]).astype(np.float32)
		emb_sparse = csr_matrix(emb)

		X = hstack([q1_tfidf, q2_tfidf, emb_sparse, num_sparse], format='csr')

		# Predict
		pred = model.predict(X)[0]
		if hasattr(model, "predict_proba"):
			proba = float(model.predict_proba(X)[0, 1])
		else:
			# fallback for models without predict_proba
			proba = float(pred)

		st.subheader("Result")
		st.write(f"Prediction: {'Duplicate' if pred == 1 else 'Not Duplicate'}")
		st.write(f"Confidence (duplicate class): {proba:.4f}")

		# Clean intermediate objects
		del num_features, num_sparse, q1_tfidf, q2_tfidf, q1_emb, q2_emb, emb, emb_sparse, X
		gc.collect()


if __name__ == '__main__':
	main()


