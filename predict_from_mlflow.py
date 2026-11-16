"""Loads whatever version holds the @champion alias and searches new data.

No Hugging Face import here: prediction goes only through MLflow.
Usage: python predict_from_mlflow.py [champion|challenger]
Author: Sai Vignesh (1MS25CI127)
"""
import json
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient

from common import REGISTERED_MODEL_NAME, TRACKING_URI

alias = sys.argv[1] if len(sys.argv) > 1 else "champion"
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
hf_id = mv.tags.get("hf_model_id") or client.get_run(mv.run_id).data.params["hf_model_id"]
uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
print(f"Loading {uri}")
print(f"  -> registered version : {mv.version}")
print(f"  -> source run         : {mv.run_id}")
print(f"  -> Hugging Face model : {hf_id}\n")

model = mlflow.pyfunc.load_model(uri)

queries = pd.read_csv("data/new_data.csv")          # 8 new queries, not in eval set
corpus = pd.read_csv("data/new_corpus.csv")         # 10 new passages written by me
inp = pd.DataFrame({"query": queries["query"],
                    "candidates": [json.dumps(corpus["passage"].tolist())] * len(queries)})
out = model.predict(inp)

pid = dict(zip(corpus["passage"], corpus["passage_id"]))
out.insert(0, "query_id", queries["query_id"])
out["predicted_passage_id"] = out["best_passage"].map(pid)
out["expected_passage_id"] = queries["expected_passage_id"]
out["correct"] = (out["predicted_passage_id"] == out["expected_passage_id"]).astype(int)
out["alias"], out["model_version"], out["hf_model_id"] = alias, mv.version, hf_id

pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 48)
print(out[["query_id", "query", "predicted_passage_id", "expected_passage_id", "best_score", "correct"]].to_string(index=False))
print(f"\nTop-1 correct: {out['correct'].sum()}/{len(out)}")
fname = "outputs/new_data_predictions.csv" if alias == "champion" else f"outputs/new_data_predictions_{alias}.csv"
out.to_csv(fname, index=False)
print(f"Saved {fname}")
