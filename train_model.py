import re
import gc
import json
import numpy as np
import pandas as pd
import nltk
from pathlib import Path
from typing import List

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	precision_score,
	recall_score,
	f1_score,
	accuracy_score,
	roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, vstack, hstack
import gensim.downloader as api
import joblib


def ensure_nltk():
	# Download minimal NLTK data required
	nltk.download('punkt', quiet=True)
	nltk.download('wordnet', quiet=True)


def clean_text_series(series: pd.Series, lemm: WordNetLemmatizer) -> pd.Series:
	def _clean(text: str) -> str:
		text = str(text).lower()
		text = re.sub(r'[^\w\s]', '', text)
		tokens = word_tokenize(text)
		return " ".join(lemm.lemmatize(t) for t in tokens)

	# Apply in chunks to control memory usage on large datasets
	chunk_size = 50000
	result = series.copy()
	for start in range(0, len(series), chunk_size):
		end = min(start + chunk_size, len(series))
		result.iloc[start:end] = series.iloc[start:end].apply(_clean)
	return result


def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
	num_df = pd.DataFrame(index=df.index)
	num_df['q1_len'] = df['question1'].str.len().astype(np.int32)
	num_df['q2_len'] = df['question2'].str.len().astype(np.int32)
	num_df['q1_num_words'] = df['question1'].apply(lambda x: len(x.split())).astype(np.int32)
	num_df['q2_num_words'] = df['question2'].apply(lambda x: len(x.split())).astype(np.int32)
	def common_words_count(q1: str, q2: str) -> int:
		return len(set(q1.split()) & set(q2.split()))
	def total_words_count(q1: str, q2: str) -> int:
		return len(set(q1.split())) + len(set(q2.split()))
	num_df['words_common'] = df.apply(lambda r: common_words_count(r['question1'], r['question2']), axis=1).astype(np.int32)
	num_df['words_total'] = df.apply(lambda r: total_words_count(r['question1'], r['question2']), axis=1).astype(np.int32)
	num_df['word_share'] = (num_df['words_common'] / num_df['words_total'].replace(0, 1)).astype(np.float32)
	return num_df


def compute_tfidf(df_q1: pd.Series, df_q2: pd.Series) -> tuple[TfidfVectorizer, csr_matrix, csr_matrix]:
	all_questions = pd.concat([df_q1, df_q2], axis=0).reset_index(drop=True)
	tfidf = TfidfVectorizer(max_features=3000, min_df=3, ngram_range=(1, 2))
	tfidf.fit(all_questions)
	q1_tfidf = tfidf.transform(df_q1)
	q2_tfidf = tfidf.transform(df_q2)
	return tfidf, q1_tfidf, q2_tfidf


def compute_tfidf_weighted_embeddings(
	df_q1: pd.Series,
	df_q2: pd.Series,
	tfidf: TfidfVectorizer,
	chunk_size: int = 50000
) -> csr_matrix:
	# Load a compact pretrained embedding model (~128MB)
	w2v_model = api.load('glove-wiki-gigaword-100')
	emb_dim = w2v_model.vector_size
	vocab = tfidf.vocabulary_
	idf = tfidf.idf_
	index_to_idf = {token: float(idf[idx]) for token, idx in vocab.items()}

	def tfidf_weighted_avg(text: str) -> np.ndarray:
		words = text.split()
		vecs: List[np.ndarray] = []
		weights: List[float] = []
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

	w2v_sparse_chunks: List[csr_matrix] = []
	for start in range(0, len(df_q1), chunk_size):
		end = min(start + chunk_size, len(df_q1))
		q1_chunk = df_q1.iloc[start:end].apply(tfidf_weighted_avg).tolist()
		q2_chunk = df_q2.iloc[start:end].apply(tfidf_weighted_avg).tolist()
		q1_arr = np.vstack(q1_chunk).astype(np.float32)
		q2_arr = np.vstack(q2_chunk).astype(np.float32)
		combined = np.hstack([q1_arr, q2_arr]).astype(np.float32)  # shape: (n_chunk, emb_dim*2)
		combined_csr = csr_matrix(combined)
		w2v_sparse_chunks.append(combined_csr)
		del q1_chunk, q2_chunk, q1_arr, q2_arr, combined, combined_csr
		gc.collect()
	w2v_sparse = vstack(w2v_sparse_chunks, format='csr')
	del w2v_sparse_chunks
	gc.collect()
	return w2v_sparse


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
	"""Find the probability threshold that maximizes F1 score."""
	thresholds = np.linspace(0.1, 0.9, 81)
	best_thresh = 0.5
	best_f1 = -1.0
	for thresh in thresholds:
		preds = (y_proba >= thresh).astype(np.int8)
		f1 = f1_score(y_true, preds)
		if f1 > best_f1:
			best_f1 = f1
			best_thresh = float(thresh)
	return best_thresh


def main():
	ensure_nltk()
	df = pd.read_csv('train.csv')
	df = df[['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']].copy()

	lemm = WordNetLemmatizer()
	df['question1'] = clean_text_series(df['question1'], lemm)
	df['question2'] = clean_text_series(df['question2'], lemm)

	num_df = build_numeric_features(df)
	tfidf, q1_tfidf, q2_tfidf = compute_tfidf(df['question1'], df['question2'])
	w2v_sparse = compute_tfidf_weighted_embeddings(df['question1'], df['question2'], tfidf)

	num_sparse = csr_matrix(num_df.values.astype(np.float32))
	X = hstack([q1_tfidf, q2_tfidf, w2v_sparse, num_sparse], format='csr')
	y = df['is_duplicate'].values.astype(np.int8)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

	model = RandomForestClassifier(
		n_estimators=600,
		max_depth=None,
		min_samples_split=2,
		min_samples_leaf=1,
		n_jobs=-1,
		random_state=42,
		class_weight='balanced',
	)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	y_proba = model.predict_proba(X_test)[:, 1]

	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	accuracy = accuracy_score(y_test, y_pred)
	roc_auc = roc_auc_score(y_test, y_proba)
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1: {f1:.4f}")
	print(f"Accuracy: {accuracy:.4f}")
	print(f"ROC-AUC: {roc_auc:.4f}")

	best_threshold = find_optimal_threshold(y_test, y_proba)
	print(f"Best probability threshold (F1 maximized): {best_threshold:.2f}")

	# Persist artifacts
	out_dir = Path('models')
	out_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, out_dir / 'model.joblib')
	joblib.dump(tfidf, out_dir / 'tfidf.joblib')

	# Save feature meta for validation in the app
	meta = {
		'x_shape': [int(X.shape[0]), int(X.shape[1])],
		'embedding_dim': 100,
		'tfidf_max_features': 3000,
		'ngram_range': [1, 2],
		'numeric_feature_count': int(num_sparse.shape[1]),
		'best_threshold': best_threshold
	}
	with open(out_dir / 'feature_meta.json', 'w', encoding='utf-8') as f:
		json.dump(meta, f)


if __name__ == '__main__':
	main()


