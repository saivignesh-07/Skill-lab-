# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student:** Sai Vignesh  **USN:** 1MS25CI127
**Course:** Skill Lab – MLOps with MLflow (Mini Project)
**Task chosen:** Semantic search (task #21 vs #22 from the brief)

---

## 1. Task and models

| Item | Value |
|---|---|
| Task | Semantic search / passage retrieval |
| Dataset | SQuAD v1.1 validation split (`rajpurkar/squad`) |
| Evaluation subset | 200 queries searched over a 500-passage corpus (seed 42) |
| Model A – current **Champion** | `sentence-transformers/all-MiniLM-L6-v2` (~22M params) |
| Model B – **Challenger** | `BAAI/bge-small-en-v1.5` (~33M params) |
| Registered model name | `semantic-search-model` |
| MLflow experiment | `1MS25CI127-hf-semantic-search-champion-challenger` |

**How the evaluation set is built.** From the SQuAD validation split I keep one question per paragraph, shuffle the paragraphs with seed 42, and take 500. The first 200 paragraphs each give one query (the SQuAD question), and the other 300 paragraphs are distractors. Each query has exactly one correct passage. Both models embed the **same queries and the same corpus in the same order** (`data/eval_inputs.csv`, `data/corpus.csv`). Retrieval uses cosine similarity on normalised embeddings.

Model-specific handling: BGE's optional query instruction (`"Represent this sentence for searching relevant passages: "`) is added to queries only, as the model card recommends. MiniLM uses no prefix.

**Metrics (higher is better unless stated):**
Recall@1, Recall@3, Recall@5, Recall@10, **MRR@10 (primary)** and nDCG@10. Mean rank is lower-is-better. `avg_latency_ms` is the time per encoded text (lower is better), `avg_query_latency_ms` is the time per query, and `total_inference_s` is the time to encode the corpus plus the queries.

---

## 2. How to run

```powershell
# 1. Environment
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Tracking server (keep this terminal open)
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000

# 3. In a second terminal: evaluate both models and log everything
python evaluate.py

# 4. Register both models and set the @champion / @challenger aliases
python register_champion.py

# 5. Load the Champion by alias and predict on new data
python predict_from_mlflow.py              # uses @champion
python predict_from_mlflow.py challenger   # stretch goal: the other alias
```

No run IDs are hard-coded. `register_champion.py` finds the latest `compare_semantic-search` run and reads the winner from its tags. `predict_from_mlflow.py` resolves the alias through the registry.

---

## 3. Champion rule (written before the final run)

1. **Primary metric:** MRR@10, where higher is better. It rewards putting the right passage at the top of the list, which matters most to a search user.
2. **Minimum improvement margin:** the Challenger must beat the Champion by **at least 0.02 absolute MRR@10**. With 200 queries, one query moving from rank 2 to rank 1 changes MRR by 0.0025, so 0.02 means about eight queries' worth of real ranking gains, not just noise from one or two queries.
3. **Guardrail:** the Challenger is rejected if its average latency is **more than 3× the Champion's**, because search runs on every user query.
4. **Tie-breaker:** if |ΔMRR@10| < 0.02, keep the **faster** model.
5. **Direction:** Recall@k, MRR@10 and nDCG@10 are higher-is-better. Latency, total time and mean rank are lower-is-better.

The rule is also hard-coded in `evaluate.py` (`PRIMARY_METRIC`, `MARGIN`, `LATENCY_GUARDRAIL`) and logged as parameters of the summary run.

---

## 4. Results

200 queries, 500-passage corpus, CPU, batch size 32, max sequence length 256.

| Metric | A: MiniLM-L6 (Champion) | B: bge-small (Challenger) | B − A |
|---|---|---|---|
| Recall@1 | 0.7950 | **0.8400** | +0.0450 |
| Recall@3 | **0.9400** | 0.9200 | −0.0200 |
| Recall@5 | **0.9650** | 0.9600 | −0.0050 |
| Recall@10 | 0.9750 | **0.9900** | +0.0150 |
| **MRR@10 (primary)** | 0.8615 | **0.8865** | **+0.0250** |
| nDCG@10 | 0.8897 | **0.9114** | +0.0217 |
| Mean rank (lower better) | **1.81** | 1.85 | +0.04 |
| Avg latency, ms per text | **29.63** | 71.94 | 2.43× |
| Avg query latency, ms | **3.85** | 11.69 | |
| Total inference, s | **20.74** | 50.36 | |

Plots (in `outputs/` and logged to MLflow under `plots/`): `metrics_comparison.png` (bar chart of every metric), `recall_at_k_curve.png`, `latency_comparison.png`, and `rank_hist_<model>.png` (per-query rank distribution).

### Qualitative examples where the models disagree
(Full list of 38 disagreements: `outputs/disagreement_examples.csv`.)

| Query | A rank | B rank | Comment |
|---|---|---|---|
| In what century did the history of the steam engine begin? | 5 | 1 | Same article. B picks the exact paragraph, while A ranks it 5th behind a sibling paragraph. |
| What is the name of contemporary Mongolian currency? | 2 | 1 | "Mongolian" pulls A to a Yuan-dynasty paragraph. B finds the Genghis Khan paragraph that names the currency. |
| What is it called when there is an active attempt to overthrow a government…? | 3 | 1 | Abstract wording. B links it to civil disobedience, while A drifts to Islamism. |
| Who is the producer of Doctor Who? | 1 | 2 | A wins here. B prefers another Doctor Who paragraph. |
| What did Luther's answer to the antinomians reaffirm? | 1 | 6 | A wins clearly. B gets lost among many Martin Luther paragraphs. |

---

## 5. Decision

**Champion: `BAAI/bge-small-en-v1.5` (Model B, registry version 2, alias `@champion`).**
`all-MiniLM-L6-v2` keeps the `@challenger` alias (version 1).

Yes, the Challenger improves on the Champion under the pre-written rule. Its MRR@10 is 0.8865 against 0.8615, a gain of +0.0250, which clears the 0.02 margin. Its latency is 2.43× the Champion's, which is inside the 3× guardrail, so the swap is allowed. The gain shows up where it matters most for search, at the top of the list: Recall@1 rises from 0.795 to 0.840 (+4.5 points), and nDCG@10 rises by +0.022. Recall@10 also improves from 0.975 to 0.990, so B misses fewer queries entirely.

The gain is real but modest, and the sample is small. On the 200 queries, B ranks the right passage first in 19 cases where A does not, and A wins 10 cases the other way. A sign test on that 19 vs 10 split gives p ≈ 0.14, and a bootstrap 95% interval for ΔMRR@10 is roughly −0.009 to +0.060. So I would not claim B is better on every dataset; a larger evaluation set would be needed to confirm it. MiniLM is also slightly better at Recall@3 and Recall@5, so the two models are close once you look past the top hit.

The trade-off is speed and size. B has about 1.5× the parameters and is roughly 2.4× slower on CPU (11.7 ms vs 3.9 ms per query). That is still well within interactive limits for a small corpus, and my rule allowed it, so B is promoted. If latency ever became the bottleneck, MiniLM is a reasonable fallback and is already registered as `@challenger`.

---

## 6. Load from MLflow and predict on new data

`predict_from_mlflow.py` never imports Hugging Face. It:
1. loads `models:/semantic-search-model@champion`;
2. prints the registered version, the source run and the Hugging Face model ID, read from the registry tags or run params;
3. searches 8 new queries (`data/new_data.csv`) over 10 new passages I wrote about Bengaluru, Karnataka, ML and MLOps (`data/new_corpus.csv`), none of which are in the SQuAD evaluation set;
4. prints a table and saves `outputs/new_data_predictions.csv`.

Result: the Champion (bge-small, v2) got 8/8 top-1 correct. **Stretch goal:** running `python predict_from_mlflow.py challenger` with the same, unedited script loads v1 (MiniLM) and also gets 8/8. Its similarity scores are noticeably lower on paraphrased queries, though, for example 0.317 vs 0.721 for "food made from finger millet". The output is saved in `outputs/new_data_predictions_challenger.csv`.

The pyfunc model (`common.SemanticSearchModel`) takes a DataFrame with `query` and `candidates` (a JSON list of passages) and returns `query`, `best_passage`, `best_score` and the `top3` matches. The Sentence-Transformers weights are stored as an MLflow artifact, so loading never needs the Hugging Face Hub.

---

## 7. What is logged in MLflow

| Run | Contents |
|---|---|
| `A_semantic-search_minilm-l6` / `B_semantic-search_bge-small` (nested) | **Params:** hf_model_id, task, role, batch_size, max_seq_length, top_k, similarity, query_prefix, eval_samples, corpus_size, seed, device. **Metrics:** all metrics in section 4. **Tags:** student_usn, dataset_name, library_versions. **Model:** pyfunc `model`. **Artifacts:** predictions CSV, metrics_summary JSON, eval_inputs.csv, rank histogram, predictions table. |
| `compare_semantic-search` (parent) | **Params:** primary_metric, margin, latency_guardrail_x, metric_direction, model_A, model_B. **Metric:** delta_primary_metric, latency_ratio_B_over_A, plus A_*/B_* copies. **Tags:** winner, winner_hf_model_id, decision_reason, run IDs. **Artifacts:** comparison CSV/JSON, disagreements, 3 plots, README. |

---

## 8. Repository structure

```
1MS25CI127-mlflow-champion-challenger/
├── README.md
├── requirements.txt
├── common.py                 # pyfunc wrapper + metric helpers + config
├── evaluate.py               # runs both models, logs everything to MLflow
├── register_champion.py      # registers versions, sets @champion/@challenger
├── predict_from_mlflow.py    # loads @champion via MLflow, predicts new data
├── data/
│   ├── eval_inputs.csv       # 200 shared evaluation queries + gold passage id
│   ├── corpus.csv            # 500 passages searched by both models
│   ├── new_data.csv          # 8 new queries (not in eval set)
│   └── new_corpus.csv        # 10 new passages
├── outputs/
│   ├── predictions_minilm-l6.csv / predictions_bge-small.csv
│   ├── metrics_summary.json (+ per-model files)
│   ├── model_comparison.csv
│   ├── metrics_comparison.png, recall_at_k_curve.png, latency_comparison.png
│   ├── rank_hist_minilm-l6.png, rank_hist_bge-small.png
│   ├── disagreement_examples.csv, qualitative_examples.csv
│   └── new_data_predictions.csv (+ _challenger)
└── screenshots/
    ├── 01_mlflow_experiment_runs.png
    ├── 02_mlflow_compare_view.png
    ├── 03_model_registry_aliases.png
    ├── 04_compare_summary_run.png
    ├── 05_predict_from_mlflow_terminal.png
    ├── 06_predict_challenger_alias.png
    └── 07_register_champion_terminal.png
```

`mlruns/`, `mlartifacts/`, `mlflow.db` and model weights are excluded through `.gitignore`.

---

## 9. Limitations and declaration

- **Sample size:** 200 queries and 500 passages. A difference of about 0.02 MRR is near the noise level, as the sign test and bootstrap in section 5 show.
- **Easy distractors:** SQuAD questions often reuse words from their paragraph, and many distractors come from unrelated articles. Real search traffic would probably be harder.
- **One correct passage per query:** a different passage that also answers the question still counts as a miss.
- **Latency** was measured on a 2-core CPU and will be different on other hardware. The ranking of the two models by speed should hold.
- **New data** is only 8 hand-written queries. That is enough to show the pipeline works, but not enough to rank the models.
- **AI assistance:** an AI assistant was used to help structure the code and draft this README. I ran every script, the reported numbers come from the actual runs, and I can explain each part in the viva.
