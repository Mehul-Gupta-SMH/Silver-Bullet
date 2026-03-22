"""
One-shot script to import existing training/test JSON reports into MLflow.
Run with: ./.sbvenv/Scripts/python import_reports_to_mlflow.py
"""

import json
import glob
import os
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow/mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


def get_or_create_experiment(name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        return client.create_experiment(name)
    return exp.experiment_id


def import_training_report(path: str):
    with open(path) as f:
        d = json.load(f)

    tp = d.get("training_parameters", {})
    tm = d.get("training_metrics", {})
    ma = d.get("model_architecture", {})
    di = d.get("dataset_info", {})
    hi = d.get("hardware_info", {})

    mode = tp.get("mode", "general")
    exp_name = f"silverbullet-{mode}"
    exp_id = get_or_create_experiment(exp_name)

    # Parse timestamp for run name
    ts = d.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        run_name = f"train-{dt.strftime('%Y%m%d-%H%M%S')}"
        start_time_ms = int(dt.timestamp() * 1000)
    except Exception:
        run_name = os.path.basename(path).replace(".json", "")
        start_time_ms = None

    # Determine if final or current (incomplete)
    is_final = "final" in path
    run_name += "-final" if is_final else "-incomplete"

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        # Params
        params = {
            "mode": mode,
            "num_epochs": tp.get("num_epochs"),
            "learning_rate": tp.get("learning_rate"),
            "batch_size": tp.get("batch_size"),
            "optimizer": tp.get("optimizer"),
            "loss_function": tp.get("loss_function"),
            "early_stopping_patience": tp.get("early_stopping_patience"),
            "early_stopping_min_delta": tp.get("early_stopping_min_delta"),
            "num_feature_channels": tp.get("num_feature_channels"),
            "model_name": ma.get("name"),
            "total_parameters": ma.get("total_parameters"),
            "device": hi.get("device"),
            "train_samples": di.get("train_samples"),
            "val_samples": di.get("val_samples"),
        }
        mlflow.log_params({k: v for k, v in params.items() if v is not None})

        # Per-epoch metrics
        epochs = tm.get("epochs", [])
        train_losses = tm.get("train_losses", [])
        val_losses = tm.get("val_losses", [])
        val_accuracies = tm.get("val_accuracies", [])

        for i, epoch in enumerate(epochs):
            step_metrics = {}
            if i < len(train_losses):
                step_metrics["train_loss"] = train_losses[i]
            if i < len(val_losses):
                step_metrics["val_loss"] = val_losses[i]
            if i < len(val_accuracies):
                step_metrics["val_accuracy"] = val_accuracies[i]
            if step_metrics:
                mlflow.log_metrics(step_metrics, step=epoch)

        # Summary metrics
        summary = {}
        if tm.get("best_val_loss") is not None:
            summary["best_val_loss"] = tm["best_val_loss"]
        if tm.get("best_val_accuracy") is not None:
            summary["best_val_accuracy"] = tm["best_val_accuracy"]
        if tm.get("best_epoch") is not None:
            summary["best_epoch"] = float(tm["best_epoch"])
        if summary:
            mlflow.log_metrics(summary)

        mlflow.set_tag("source_file", os.path.basename(path))
        mlflow.set_tag("report_type", "final" if is_final else "incomplete")

        print(f"  [OK] Imported training run: {run_name} (experiment: {exp_name})")
        return run.info.run_id


def import_test_report(path: str):
    with open(path) as f:
        d = json.load(f)

    metrics = d.get("metrics", {})
    di = d.get("dataset_info", {})
    tp = d.get("test_parameters", {})
    hi = d.get("hardware_info", {})
    mi = d.get("model_info", {})

    # Test reports don't have mode info — put in a general test experiment
    mode = di.get("mode", "general")
    exp_name = f"silverbullet-{mode}-tests"
    exp_id = get_or_create_experiment(exp_name)

    ts = d.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        run_name = f"test-{dt.strftime('%Y%m%d-%H%M%S')}"
    except Exception:
        run_name = os.path.basename(path).replace(".json", "")

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.log_params({
            "threshold": tp.get("threshold"),
            "batch_size": tp.get("batch_size"),
            "device": hi.get("device") or tp.get("device"),
            "test_samples": di.get("test_samples"),
            "feature_dimension": di.get("feature_dimension"),
            "model_name": mi.get("name"),
            "total_parameters": mi.get("total_parameters"),
        })

        test_metrics = {}
        if metrics.get("accuracy") is not None:
            test_metrics["test_accuracy"] = metrics["accuracy"]
        if metrics.get("roc_auc") is not None:
            test_metrics["test_roc_auc"] = metrics["roc_auc"]
        if metrics.get("average_precision") is not None:
            test_metrics["test_avg_precision"] = metrics["average_precision"]
        if test_metrics:
            mlflow.log_metrics(test_metrics)

        mlflow.set_tag("source_file", os.path.basename(path))
        mlflow.set_tag("report_type", "test")

        print(f"  [OK] Imported test run: {run_name} (experiment: {exp_name})")
        return run.info.run_id


def main():
    print("Importing training reports...")
    training_files = sorted(glob.glob("training_reports/training_report_*.json"))
    for path in training_files:
        try:
            import_training_report(path)
        except Exception as e:
            print(f"  [FAIL] {path}: {e}")

    print("\nImporting test reports...")
    test_files = sorted(glob.glob("test_reports/test_report_*.json"))
    for path in test_files:
        try:
            import_test_report(path)
        except Exception as e:
            print(f"  [FAIL] {path}: {e}")

    print("\nDone. Open http://127.0.0.1:5000 to view experiments.")



if __name__ == "__main__":
    main()
