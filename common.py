"""Shared helpers: the MLflow pyfunc wrapper and retrieval metrics.

Author: Sai Vignesh (1MS25CI127)
"""
import json
import time

import mlflow.pyfunc
import numpy as np
import pandas as pd

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "1MS25CI127-hf-semantic-search-champion-challenger"
REGISTERED_MODEL_NAME = "semantic-search-model"
TASK = "semantic-search"
SEED = 42
K_VALUES = [1, 3, 5, 10]


class SemanticSearchModel(mlflow.pyfunc.PythonModel):
    """Wraps a Sentence-Transformers bi-encoder as an MLflow pyfunc model.

    Input  : DataFrame with columns
             - query      : the search query (str)
             - candidates : JSON list of passages to search in (str)
    Output : DataFrame with query, best_passage, best_score, top3 (JSON)
    The Hugging Face model is loaded from the MLflow artifact, so callers
    never touch Hugging Face code directly.
    """

    def __init__(self, query_prefix="", passage_prefix=""):
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(context.artifacts["hf_model_dir"], device="cpu")

    def predict(self, context, model_input, params=None):
        rows = []
        for _, r in model_input.iterrows():
            passages = json.loads(r["candidates"])
            q = self.model.encode([self.query_prefix + r["query"]], normalize_embeddings=True)
            p = self.model.encode([self.passage_prefix + x for x in passages], normalize_embeddings=True)
            scores = (p @ q.T).ravel()
            order = np.argsort(-scores)
            rows.append({
                "query": r["query"],
                "best_passage": passages[order[0]],
                "best_score": round(float(scores[order[0]]), 4),
                "top3": json.dumps([{"passage": passages[i][:80], "score": round(float(scores[i]), 4)} for i in order[:3]]),
            })
        return pd.DataFrame(rows)


def retrieval_metrics(ranks, k_values=K_VALUES):
    """ranks: 1-based rank of the correct passage for each query."""
    ranks = np.asarray(ranks)
    m = {f"recall_at_{k}": float(np.mean(ranks <= k)) for k in k_values}
    m["mrr_at_10"] = float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0)))
    m["ndcg_at_10"] = float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)))
    m["mean_rank"] = float(np.mean(ranks))
    return m


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t
