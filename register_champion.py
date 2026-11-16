"""Registers both model runs and sets the champion / challenger aliases.

Reads the latest compare_semantic-search run from MLflow, so no run IDs
are hard-coded.
Author: Sai Vignesh (1MS25CI127)
"""
import mlflow
from mlflow import MlflowClient

from common import EXPERIMENT_NAME, REGISTERED_MODEL_NAME, TASK, TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
compare = client.search_runs([exp.experiment_id], f"attributes.run_name = 'compare_{TASK}'",
                             order_by=["attributes.start_time DESC"], max_results=1)[0]
winner = compare.data.tags["winner"]
run_ids = {"A": compare.data.tags["run_A_id"], "B": compare.data.tags["run_B_id"]}
print(f"Compare run {compare.info.run_id}: winner = {winner} ({compare.data.tags['winner_hf_model_id']})")

try:
    client.create_registered_model(REGISTERED_MODEL_NAME,
                                   description="1MS25CI127 semantic search Champion-Challenger (MiniLM vs bge-small)")
except mlflow.exceptions.MlflowException:
    pass  # already exists

versions = {}
for key, rid in run_ids.items():
    hf_id = client.get_run(rid).data.params["hf_model_id"]
    mv = mlflow.register_model(f"runs:/{rid}/model", REGISTERED_MODEL_NAME,
                               tags={"hf_model_id": hf_id, "model_label": key, "student_usn": "1MS25CI127"})
    versions[key] = mv.version
    print(f"Registered {key} ({hf_id}) as version {mv.version}")

loser = "B" if winner == "A" else "A"
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", versions[winner])
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", versions[loser])
client.set_model_version_tag(REGISTERED_MODEL_NAME, versions[winner], "role", "champion")
client.set_model_version_tag(REGISTERED_MODEL_NAME, versions[loser], "role", "challenger")
print(f"@champion -> v{versions[winner]}   @challenger -> v{versions[loser]}")
