"""Champion-Challenger evaluation for semantic search.

Model A (current Champion) : sentence-transformers/all-MiniLM-L6-v2
Model B (Challenger)       : BAAI/bge-small-en-v1.5

Both models embed the SAME 200 queries and the SAME 500-passage corpus
(data/eval_inputs.csv, data/corpus.csv). For every query we rank the
corpus by cosine similarity and record where the correct passage lands.

Author: Sai Vignesh (1MS25CI127)
"""
import json
import os
import platform
import random
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer

from common import (EXPERIMENT_NAME, K_VALUES, SEED, TASK, TRACKING_URI,
                    SemanticSearchModel, retrieval_metrics, timed)

N_QUERIES = 200
N_DISTRACTORS = 300
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256
DATASET_NAME = "rajpurkar/squad (validation)"

# Champion rule (written in README.md BEFORE the final run - do not edit)
PRIMARY_METRIC = "mrr_at_10"
MARGIN = 0.02            # absolute MRR@10 the Challenger must gain
LATENCY_GUARDRAIL = 3.0  # Challenger rejected if > 3x Champion latency

MODELS = {
    "A": {"run_name": f"A_{TASK}_minilm-l6", "hf_model_id": "sentence-transformers/all-MiniLM-L6-v2",
          "short": "minilm-l6", "role": "champion", "query_prefix": "", "passage_prefix": ""},
    "B": {"run_name": f"B_{TASK}_bge-small", "hf_model_id": "BAAI/bge-small-en-v1.5",
          "short": "bge-small", "role": "challenger",
          "query_prefix": "Represent this sentence for searching relevant passages: ", "passage_prefix": ""},
}

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def build_eval_set():
    """Fixed, seeded subset of SQuAD v1.1 validation (one question per paragraph)."""
    if os.path.exists("data/eval_inputs.csv") and os.path.exists("data/corpus.csv"):
        return pd.read_csv("data/eval_inputs.csv"), pd.read_csv("data/corpus.csv")
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation").to_pandas()
    first_q = ds.drop_duplicates("context").reset_index(drop=True)
    idx = np.random.RandomState(SEED).permutation(len(first_q))
    chosen = first_q.iloc[idx[:N_QUERIES + N_DISTRACTORS]].reset_index(drop=True)
    corpus = pd.DataFrame({"passage_id": [f"P{i:03d}" for i in range(len(chosen))],
                           "title": chosen["title"], "passage": chosen["context"]})
    evalq = pd.DataFrame({"query_id": [f"Q{i:03d}" for i in range(N_QUERIES)],
                          "query": chosen["question"][:N_QUERIES],
                          "relevant_passage_id": corpus["passage_id"][:N_QUERIES],
                          "source_squad_id": chosen["id"][:N_QUERIES]})
    evalq.to_csv("data/eval_inputs.csv", index=False)
    corpus.to_csv("data/corpus.csv", index=False)
    return evalq, corpus


def evaluate_model(cfg, evalq, corpus):
    model = SentenceTransformer(cfg["hf_model_id"], device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH
    p_emb, t_corpus = timed(model.encode, [cfg["passage_prefix"] + p for p in corpus["passage"]],
                            batch_size=BATCH_SIZE, normalize_embeddings=True)
    q_emb, t_query = timed(model.encode, [cfg["query_prefix"] + q for q in evalq["query"]],
                           batch_size=BATCH_SIZE, normalize_embeddings=True)
    scores = q_emb @ p_emb.T
    order = np.argsort(-scores, axis=1)
    pid = corpus["passage_id"].to_numpy()
    gold = evalq["relevant_passage_id"].to_numpy()
    ranks = np.array([int(np.where(pid[order[i]] == gold[i])[0][0]) + 1 for i in range(len(evalq))])
    m = retrieval_metrics(ranks)
    total = t_corpus + t_query
    m["total_inference_s"] = total
    m["avg_latency_ms"] = 1000 * total / (len(evalq) + len(corpus))   # per encoded text
    m["avg_query_latency_ms"] = 1000 * t_query / len(evalq)
    preds = evalq.copy()
    preds["reference_passage"] = [corpus.set_index("passage_id").loc[g, "passage"][:150] for g in gold]
    preds["top1_passage_id"] = pid[order[:, 0]]
    preds["top1_passage"] = [corpus["passage"].iloc[j][:150] for j in order[:, 0]]
    preds["top1_score"] = scores[np.arange(len(evalq)), order[:, 0]].round(4)
    preds["rank_of_relevant"] = ranks
    preds["top1_correct"] = (ranks == 1).astype(int)
    return model, m, preds, ranks


def log_model_run(key, cfg, model, m, preds, evalq):
    with mlflow.start_run(run_name=cfg["run_name"], nested=True) as run:
        mlflow.log_params({"hf_model_id": cfg["hf_model_id"], "task": TASK,
                           "role": f"{cfg['role']} candidate", "batch_size": BATCH_SIZE,
                           "max_seq_length": MAX_SEQ_LENGTH, "top_k": max(K_VALUES),
                           "similarity": "cosine (normalised dot product)",
                           "query_prefix": cfg["query_prefix"] or "none",
                           "eval_samples": len(evalq), "corpus_size": N_QUERIES + N_DISTRACTORS,
                           "seed": SEED, "device": "cpu"})
        mlflow.log_metrics(m)
        mlflow.set_tags({"student_usn": "1MS25CI127", "student_name": "Sai Vignesh",
                         "dataset_name": DATASET_NAME, "model_label": key,
                         "library_versions": f"transformers={transformers.__version__}, torch={torch.__version__}, "
                                             f"mlflow={mlflow.__version__}, sentence-transformers={sentence_transformers.__version__}",
                         "python": platform.python_version()})
        out = f"outputs/predictions_{cfg['short']}.csv"
        preds.to_csv(out, index=False)
        summ = f"outputs/metrics_summary_{cfg['short']}.json"
        json.dump({"model": cfg["hf_model_id"], **m}, open(summ, "w"), indent=2)
        mlflow.log_artifact(out, "predictions")
        mlflow.log_artifact(summ, "metrics")
        mlflow.log_artifact("data/eval_inputs.csv", "data")
        mlflow.log_table(preds.head(50), "predictions_preview.json")
        # rank histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.clip(preds["rank_of_relevant"], 1, 21), bins=range(1, 23), color="#3b6ea5", edgecolor="white")
        ax.set_xlabel("Rank of relevant passage (21 = beyond 20)"); ax.set_ylabel("Queries")
        ax.set_title(f"{cfg['short']}: rank distribution (lower is better)")
        fig.tight_layout(); fig.savefig(f"outputs/rank_hist_{cfg['short']}.png", dpi=130)
        mlflow.log_figure(fig, f"plots/rank_hist_{cfg['short']}.png"); plt.close(fig)
        # model
        with tempfile.TemporaryDirectory() as d:
            mdir = os.path.join(d, "hf_model"); model.save(mdir)
            sample = pd.DataFrame({"query": ["Who won Super Bowl 50?"],
                                   "candidates": [json.dumps(["The Denver Broncos won Super Bowl 50.",
                                                              "Paris is the capital of France."])]})
            mlflow.pyfunc.log_model(name="model",
                                    python_model=SemanticSearchModel(cfg["query_prefix"], cfg["passage_prefix"]),
                                    artifacts={"hf_model_dir": mdir}, code_paths=["common.py"],
                                    input_example=sample,
                                    pip_requirements=["sentence-transformers", "torch", "pandas", "numpy", "mlflow"],
                                    metadata={"hf_model_id": cfg["hf_model_id"], "task": TASK})
        return run.info.run_id


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    evalq, corpus = build_eval_set()
    results = {}
    with mlflow.start_run(run_name=f"compare_{TASK}") as parent:
        for key, cfg in MODELS.items():
            print(f"Evaluating {key}: {cfg['hf_model_id']}")
            model, m, preds, ranks = evaluate_model(cfg, evalq, corpus)
            run_id = log_model_run(key, cfg, model, m, preds, evalq)
            results[key] = {"cfg": cfg, "metrics": m, "preds": preds, "ranks": ranks, "run_id": run_id}
            print(json.dumps({k: round(v, 4) for k, v in m.items()}, indent=2))

        A, B = results["A"]["metrics"], results["B"]["metrics"]
        delta = B[PRIMARY_METRIC] - A[PRIMARY_METRIC]
        ratio = B["avg_latency_ms"] / A["avg_latency_ms"]
        if ratio > LATENCY_GUARDRAIL:
            winner, reason = "A", f"Challenger latency {ratio:.2f}x exceeds {LATENCY_GUARDRAIL}x guardrail"
        elif delta >= MARGIN:
            winner, reason = "B", f"Challenger beats Champion by {delta:.4f} >= margin {MARGIN}"
        elif delta <= -MARGIN:
            winner, reason = "A", f"Challenger is worse by {-delta:.4f}"
        else:
            winner = "A" if A["avg_latency_ms"] <= B["avg_latency_ms"] else "B"
            reason = f"|delta| {abs(delta):.4f} < margin {MARGIN}; faster model kept"
        wcfg = results[winner]["cfg"]
        print(f"\nDecision: winner = {winner} ({wcfg['hf_model_id']}) - {reason}")

        mlflow.log_params({"primary_metric": PRIMARY_METRIC, "margin": MARGIN,
                           "latency_guardrail_x": LATENCY_GUARDRAIL, "metric_direction": "higher is better",
                           "model_A": MODELS["A"]["hf_model_id"], "model_B": MODELS["B"]["hf_model_id"],
                           "eval_samples": len(evalq)})
        mlflow.log_metrics({"delta_primary_metric": delta, "latency_ratio_B_over_A": ratio,
                            **{f"A_{k}": v for k, v in A.items()}, **{f"B_{k}": v for k, v in B.items()}})
        mlflow.set_tags({"winner": winner, "winner_hf_model_id": wcfg["hf_model_id"],
                         "winner_run_id": results[winner]["run_id"], "decision_reason": reason,
                         "run_A_id": results["A"]["run_id"], "run_B_id": results["B"]["run_id"],
                         "student_usn": "1MS25CI127", "dataset_name": DATASET_NAME})

        # comparison table + summary json
        rows = [{"metric": k, "A_minilm-l6": A[k], "B_bge-small": B[k], "difference_B_minus_A": B[k] - A[k]} for k in A]
        comp = pd.DataFrame(rows); comp.to_csv("outputs/model_comparison.csv", index=False)
        json.dump({"A": A, "B": B, "primary_metric": PRIMARY_METRIC, "margin": MARGIN,
                   "delta_primary_metric": delta, "latency_ratio": ratio, "winner": winner,
                   "winner_hf_model_id": wcfg["hf_model_id"], "reason": reason},
                  open("outputs/metrics_summary.json", "w"), indent=2)

        # plot 1: bar chart of every quality metric
        qm = [f"recall_at_{k}" for k in K_VALUES] + ["mrr_at_10", "ndcg_at_10"]
        x = np.arange(len(qm)); w = 0.38
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, [A[k] for k in qm], w, label="A: MiniLM-L6 (Champion)", color="#3b6ea5")
        ax.bar(x + w / 2, [B[k] for k in qm], w, label="B: bge-small (Challenger)", color="#e08a2c")
        for i, k in enumerate(qm):
            ax.text(i - w / 2, A[k] + 0.01, f"{A[k]:.3f}", ha="center", fontsize=7)
            ax.text(i + w / 2, B[k] + 0.01, f"{B[k]:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(qm, rotation=15); ax.set_ylim(0, 1.08)
        ax.set_title("Champion vs Challenger on every metric (higher is better)"); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig("outputs/metrics_comparison.png", dpi=130)
        mlflow.log_figure(fig, "plots/metrics_comparison.png"); plt.close(fig)

        # plot 2: Recall@k curve
        ks = list(range(1, 21))
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for key, col in (("A", "#3b6ea5"), ("B", "#e08a2c")):
            r = results[key]["ranks"]
            ax.plot(ks, [np.mean(r <= k) for k in ks], marker="o", ms=3, color=col,
                    label=f"{key}: {results[key]['cfg']['short']}")
        ax.set_xlabel("k"); ax.set_ylabel("Recall@k"); ax.set_xticks(ks[::2] + [20])
        ax.set_title("Recall@k curve (500-passage corpus)"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig("outputs/recall_at_k_curve.png", dpi=130)
        mlflow.log_figure(fig, "plots/recall_at_k_curve.png"); plt.close(fig)

        # plot 3: latency
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["A: MiniLM-L6", "B: bge-small"], [A["avg_latency_ms"], B["avg_latency_ms"]], color=["#3b6ea5", "#e08a2c"])
        ax.set_ylabel("ms per encoded text (lower is better)"); ax.set_title("Average inference latency (CPU)")
        fig.tight_layout(); fig.savefig("outputs/latency_comparison.png", dpi=130)
        mlflow.log_figure(fig, "plots/latency_comparison.png"); plt.close(fig)

        # disagreements: queries where exactly one model ranks the right passage first
        pa, pb = results["A"]["preds"], results["B"]["preds"]
        dis = pd.DataFrame({"query_id": pa["query_id"], "query": pa["query"],
                            "relevant_passage_id": pa["relevant_passage_id"],
                            "A_top1": pa["top1_passage_id"], "A_rank": pa["rank_of_relevant"],
                            "B_top1": pb["top1_passage_id"], "B_rank": pb["rank_of_relevant"]})
        dis = dis[dis["A_top1"] != dis["B_top1"]].copy()
        dis["comment"] = ""
        dis.to_csv("outputs/disagreement_examples.csv", index=False)
        for f in ["outputs/model_comparison.csv", "outputs/metrics_summary.json",
                  "outputs/disagreement_examples.csv", "data/eval_inputs.csv", "data/corpus.csv"]:
            mlflow.log_artifact(f, "comparison")
        if os.path.exists("README.md"):
            mlflow.log_artifact("README.md")
        print(comp.round(4).to_string(index=False))
        print(f"\nDisagreements: {len(dis)} queries. compare run id: {parent.info.run_id}")


if __name__ == "__main__":
    main()
