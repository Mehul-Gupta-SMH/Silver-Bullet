import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
from backend.model import TextSimilarityCNN, TextSimilarityCNNLegacy
from backend.predict import _load_model_from_checkpoint
from backend.train import TextSimilarityDataset, load_json_data
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss,
    matthews_corrcoef, cohen_kappa_score,
    f1_score, precision_score, recall_score,
)
from scipy.stats import ks_2samp
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

class TestReport:
    def __init__(self, model, test_params: dict, output_dir='test_reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self._get_model_info(model),
            'test_parameters': test_params,
            'metrics': {},
            'predictions': [],
            'dataset_info': {},
            'hardware_info': self._get_hardware_info()
        }

    def _get_model_info(self, model):
        return {
            'name': model.__class__.__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'architecture': str(model)
        }

    def _get_hardware_info(self):
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }

    def update_dataset_info(self, test_size: int, feature_dim: int):
        self.report_data['dataset_info'] = {
            'test_samples': test_size,
            'feature_dimension': feature_dim,
            'cached_features_used': True
        }

    def update_metrics(self, metrics: dict):
        self.report_data['metrics'] = metrics

    def add_prediction(self, text1: str, text2: str, true_label: int, predicted: int, probability: float):
        self.report_data['predictions'].append({
            'text1': text1,
            'text2': text2,
            'true_label': int(true_label),
            'predicted_label': int(predicted),
            'probability': float(probability)
        })

    def save_report(self):
        """Save the test report"""
        report_file = os.path.join(self.output_dir, f'test_report_{self.timestamp}.json')

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2)

        # Generate markdown summary
        self._generate_markdown_summary(report_file)

        return report_file

    def _generate_markdown_summary(self, json_report_path):
        """Generate a markdown summary of the test report"""
        md_content = f"""# Test Report Summary
Generated on: {self.report_data['timestamp']}

## Model Information
- Model Name: {self.report_data['model_info']['name']}
- Total Parameters: {self.report_data['model_info']['total_parameters']:,}
- Trainable Parameters: {self.report_data['model_info']['trainable_parameters']:,}

## Test Dataset Information
- Test Samples: {self.report_data['dataset_info']['test_samples']}
- Feature Dimension: {self.report_data['dataset_info']['feature_dimension']}
- Using Cached Features: {self.report_data['dataset_info']['cached_features_used']}

## Hardware Information
- Device: {self.report_data['hardware_info']['device']}
"""
        if self.report_data['hardware_info']['cuda_device']:
            md_content += f"- GPU: {self.report_data['hardware_info']['cuda_device']}\n"

        md_content += "\n## Test Metrics\n"
        m = self.report_data['metrics']

        md_content += "\n### Core\n"
        md_content += f"| Metric | Value |\n|--------|-------|\n"
        md_content += f"| Accuracy | {m['accuracy']:.4f} |\n"
        md_content += f"| MCC | {m['mcc']:.4f} |\n"
        md_content += f"| Cohen's Kappa | {m['cohen_kappa']:.4f} |\n"

        md_content += "\n### Ranking\n"
        md_content += f"| Metric | Value |\n|--------|-------|\n"
        md_content += f"| ROC-AUC | {m['roc_auc']:.4f} |\n"
        md_content += f"| AUPRC | {m['auprc']:.4f} |\n"
        md_content += f"| KS Statistic | {m['score_distribution']['ks_statistic']:.4f} |\n"
        md_content += f"| KS p-value | {m['score_distribution']['ks_p_value']:.4f} |\n"

        md_content += "\n### Calibration\n"
        md_content += f"| Metric | Value |\n|--------|-------|\n"
        md_content += f"| Brier Score | {m['brier_score']:.4f} |\n"
        md_content += f"| Log Loss | {m['log_loss']:.4f} |\n"

        ot = m['optimal_threshold']
        md_content += "\n### Optimal Threshold (Youden's J)\n"
        md_content += f"| Metric | Value |\n|--------|-------|\n"
        md_content += f"| Threshold | {ot['value']:.4f} |\n"
        md_content += f"| F1 | {ot['f1']:.4f} |\n"
        md_content += f"| Precision | {ot['precision']:.4f} |\n"
        md_content += f"| Recall | {ot['recall']:.4f} |\n"

        sd = m['score_distribution']
        md_content += "\n### Score Distribution\n"
        md_content += f"| | Positive class | Negative class |\n|--|--|--|\n"
        md_content += f"| Mean | {sd['positive_class']['mean']:.4f} | {sd['negative_class']['mean']:.4f} |\n"
        md_content += f"| Std  | {sd['positive_class']['std']:.4f}  | {sd['negative_class']['std']:.4f}  |\n"
        md_content += f"| Min  | {sd['positive_class']['min']:.4f}  | {sd['negative_class']['min']:.4f}  |\n"
        md_content += f"| Max  | {sd['positive_class']['max']:.4f}  | {sd['negative_class']['max']:.4f}  |\n"

        if m.get('confidence_intervals'):
            ci_level = int(next(iter(m['confidence_intervals'].values()))['ci'] * 100)
            md_content += f"\n### Bootstrap Confidence Intervals ({ci_level}%, n=1000)\n"
            md_content += f"| Metric | Point estimate | Lower | Upper | Width |\n"
            md_content += f"|--------|---------------|-------|-------|-------|\n"
            for k, v in m['confidence_intervals'].items():
                width = v['upper'] - v['lower']
                md_content += f"| {k} | {v['point']:.4f} | {v['lower']:.4f} | {v['upper']:.4f} | {width:.4f} |\n"

        if m.get('mc_dropout'):
            mc = m['mc_dropout']
            md_content += f"\n### MC Dropout Prediction Intervals ({int(mc['ci']*100)}%, {mc['n_passes']} passes)\n"
            md_content += f"| Metric | Value |\n|--------|-------|\n"
            md_content += f"| Mean probability std (epistemic uncertainty) | {mc['mean_std']:.4f} |\n"
            md_content += f"| Mean interval width | {mc['mean_interval_width']:.4f} |\n"
            md_content += f"| Fraction with std > 0.1 (high uncertainty) | {mc['high_uncertainty_frac']:.4f} |\n"

        md_content += "\n### Classification Report\n```\n"
        md_content += m['classification_report_str']
        md_content += "\n```\n"

        md_content += "\n### Confusion Matrix\n```\n"
        md_content += str(np.array(m['confusion_matrix']))
        md_content += "\n```\n"

        md_path = Path(json_report_path).with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

def _bootstrap_metric_cis(
    binary_true: np.ndarray,
    probs_flat: np.ndarray,
    predictions: np.ndarray,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Percentile bootstrap confidence intervals for all scalar test metrics.

    Returns a dict keyed by metric name, each value:
        {'point': float, 'lower': float, 'upper': float, 'ci': float}
    """
    rng = np.random.default_rng(seed)
    n = len(binary_true)
    alpha = 1.0 - ci
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

    boot: dict[str, list] = {k: [] for k in [
        'accuracy', 'roc_auc', 'auprc', 'brier_score',
        'log_loss', 'mcc', 'cohen_kappa', 'f1',
    ]}

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        bt, pb, pd = binary_true[idx], probs_flat[idx], predictions[idx]

        # Skip samples with only one class present (breaks AUC metrics)
        if len(np.unique(bt)) < 2:
            continue

        boot['accuracy'].append(float((pd == bt).mean()))
        boot['brier_score'].append(float(brier_score_loss(bt, pb)))
        boot['log_loss'].append(float(log_loss(bt, pb)))
        boot['mcc'].append(float(matthews_corrcoef(bt, pd)))
        boot['cohen_kappa'].append(float(cohen_kappa_score(bt, pd)))
        boot['f1'].append(float(f1_score(bt, pd)))
        try:
            boot['roc_auc'].append(float(roc_auc_score(bt, pb)))
        except Exception:
            pass
        try:
            boot['auprc'].append(float(average_precision_score(bt, pb)))
        except Exception:
            pass

    result = {}
    for key, values in boot.items():
        if not values:
            continue
        arr = np.array(values)
        result[key] = {
            'point': float(arr.mean()),
            'lower': float(np.percentile(arr, lo)),
            'upper': float(np.percentile(arr, hi)),
            'ci':    ci,
        }
    return result


def _mc_dropout_prediction_intervals(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device,
    n_passes: int = 50,
    ci: float = 0.95,
) -> dict:
    """
    MC Dropout prediction intervals: run n_passes stochastic forward passes
    with dropout active, collect per-sample probability distribution.

    Returns dict with per-sample arrays (length N):
        mean, std, ci_lower, ci_upper, epistemic_uncertainty
    """
    alpha = 1.0 - ci

    # Keep BN in eval (uses running stats), enable Dropout stochasticity
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    is_legacy = isinstance(model, TextSimilarityCNNLegacy)
    all_pass_probs = []  # [n_passes, N]

    with torch.no_grad():
        for _ in range(n_passes):
            pass_probs = []
            for features, lengths, _ in test_loader:
                features = features.to(device)
                lengths  = lengths.to(device)
                if is_legacy:
                    n_maps = model.fc_reduce1.in_features // (64 * 64)
                    features = features[:, :n_maps, :, :].view(features.size(0), -1)
                    out = model(features).cpu().numpy().flatten()
                else:
                    out = model(features, lengths).cpu().numpy().flatten()
                pass_probs.append(out)
            all_pass_probs.append(np.concatenate(pass_probs))

    # Shape: [n_passes, N] → [N, n_passes]
    samples = np.stack(all_pass_probs, axis=1)   # [N, n_passes]

    mean  = samples.mean(axis=1)
    std   = samples.std(axis=1)
    ci_lo = np.percentile(samples, alpha / 2 * 100, axis=1)
    ci_hi = np.percentile(samples, (1 - alpha / 2) * 100, axis=1)

    # Restore model to pure eval
    model.eval()

    return {
        'n_passes':             n_passes,
        'ci':                   ci,
        'mean_probability':     mean.tolist(),
        'std_probability':      std.tolist(),
        'ci_lower':             ci_lo.tolist(),
        'ci_upper':             ci_hi.tolist(),
        'mean_interval_width':  float((ci_hi - ci_lo).mean()),
        'mean_std':             float(std.mean()),
        'high_uncertainty_frac': float((std > 0.1).mean()),  # fraction with std > 0.1
    }


def _log_test_to_mlflow(metrics: dict, mode: str | None, report_path: str) -> None:
    """Log test metrics to MLflow under the same experiment as training. Fails silently."""
    try:
        import mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(uri)
        exp_name = mode or "general"
        mlflow.set_experiment(exp_name)
        run_name = f"{exp_name}-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics({
                "test_accuracy":           float(metrics["accuracy"]),
                "test_roc_auc":            float(metrics["roc_auc"]),
                "test_auprc":              float(metrics["auprc"]),
                "test_brier_score":        float(metrics["brier_score"]),
                "test_log_loss":           float(metrics["log_loss"]),
                "test_mcc":                float(metrics["mcc"]),
                "test_cohen_kappa":        float(metrics["cohen_kappa"]),
                "test_ks_statistic":       float(metrics["score_distribution"]["ks_statistic"]),
                "test_optimal_threshold":  float(metrics["optimal_threshold"]["value"]),
                "test_f1_at_optimal":      float(metrics["optimal_threshold"]["f1"]),
                "test_precision_at_optimal": float(metrics["optimal_threshold"]["precision"]),
                "test_recall_at_optimal":  float(metrics["optimal_threshold"]["recall"]),
                # Legacy key
                "test_avg_precision":      float(metrics["auprc"]),
                # Bootstrap CI widths (upper - lower)
                **{
                    f"test_{k}_ci_width": float(v["upper"] - v["lower"])
                    for k, v in metrics.get("confidence_intervals", {}).items()
                },
                # MC Dropout uncertainty summary
                "test_mc_mean_std":              float(metrics["mc_dropout"]["mean_std"]),
                "test_mc_mean_interval_width":   float(metrics["mc_dropout"]["mean_interval_width"]),
                "test_mc_high_uncertainty_frac": float(metrics["mc_dropout"]["high_uncertainty_frac"]),
            })
            if report_path and os.path.exists(report_path):
                mlflow.log_artifact(report_path)
        print(f"[MLflow] test metrics logged to experiment '{exp_name}'")
    except Exception as exc:
        print(f"[MLflow] test logging skipped ({exc})")


def test_model(model, test_loader, test_pairs, device='cuda'):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []

    # Initialize test report
    test_params = {
        'threshold': 0.5,
        'batch_size': test_loader.batch_size,
        'device': device
    }
    report = TestReport(model, test_params)

    # Update dataset information
    report.update_dataset_info(
        test_size=len(test_loader.dataset),
        feature_dim=test_loader.dataset.features.shape[1]
    )

    is_legacy = isinstance(model, TextSimilarityCNNLegacy)

    with torch.no_grad():
        for i, (features, lengths, labels) in enumerate(test_loader):
            features = features.to(device)
            lengths  = lengths.to(device)
            labels   = labels.to(device)
            if is_legacy:
                trained_num_maps = model.fc_reduce1.in_features // (64 * 64)
                features = features[:, :trained_num_maps, :, :]
                features = features.view(features.size(0), -1)
                outputs = model(features)
            else:
                outputs = model(features, lengths)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float()

            # Store predictions and true labels
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs)

            # Add individual predictions to report
            for j in range(len(preds)):
                report.add_prediction(
                    text1=test_pairs[i * test_loader.batch_size + j][0],
                    text2=test_pairs[i * test_loader.batch_size + j][1],
                    true_label=labels[j].item(),
                    predicted=preds[j].item(),
                    probability=probs[j][0]
                )

    # Convert to numpy arrays (flattened to 1-D; labels may be [N,1] from DataLoader)
    predictions   = np.array(predictions)
    true_labels   = np.array(true_labels)
    probabilities = np.array(probabilities)

    # Binarise true labels at 0.5 so sklearn metrics work with both binary (0/1)
    # and continuous float labels (e.g. 0.7 faithfulness scores).
    binary_true = (np.array(true_labels).flatten() >= 0.5).astype(int)
    predictions = np.array(predictions).flatten().astype(int)
    probs_flat  = np.array(probabilities).flatten()

    # ── Core classification metrics ──────────────────────────────────────────
    conf_matrix  = confusion_matrix(binary_true, predictions)
    class_report = classification_report(binary_true, predictions, target_names=['Not Similar', 'Similar'])
    accuracy     = float((predictions == binary_true).mean())

    # ── Ranking / scoring metrics ─────────────────────────────────────────────
    roc_auc    = float(roc_auc_score(binary_true, probs_flat))
    auprc      = float(average_precision_score(binary_true, probs_flat))   # area under PR curve

    fpr, tpr, roc_thresholds = roc_curve(binary_true, probs_flat)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(binary_true, probs_flat)

    # ── Calibration metrics ───────────────────────────────────────────────────
    brier = float(brier_score_loss(binary_true, probs_flat))
    logloss = float(log_loss(binary_true, probs_flat))

    # ── Optimal threshold (Youden's J: max TPR - FPR on ROC curve) ───────────
    youden_idx       = int(np.argmax(tpr - fpr))
    optimal_threshold = float(roc_thresholds[youden_idx])
    preds_optimal    = (probs_flat >= optimal_threshold).astype(int)
    f1_optimal       = float(f1_score(binary_true, preds_optimal))
    precision_optimal = float(precision_score(binary_true, preds_optimal))
    recall_optimal   = float(recall_score(binary_true, preds_optimal))

    # ── Agreement metrics ─────────────────────────────────────────────────────
    mcc   = float(matthews_corrcoef(binary_true, predictions.flatten().astype(int)))
    kappa = float(cohen_kappa_score(binary_true, predictions.flatten().astype(int)))

    # ── Score distribution per class ──────────────────────────────────────────
    pos_scores = probs_flat[binary_true == 1]
    neg_scores = probs_flat[binary_true == 0]
    ks_stat, ks_p = ks_2samp(pos_scores, neg_scores)

    score_distribution = {
        'positive_class': {
            'mean': float(pos_scores.mean()) if len(pos_scores) else None,
            'std':  float(pos_scores.std())  if len(pos_scores) else None,
            'min':  float(pos_scores.min())  if len(pos_scores) else None,
            'max':  float(pos_scores.max())  if len(pos_scores) else None,
        },
        'negative_class': {
            'mean': float(neg_scores.mean()) if len(neg_scores) else None,
            'std':  float(neg_scores.std())  if len(neg_scores) else None,
            'min':  float(neg_scores.min())  if len(neg_scores) else None,
            'max':  float(neg_scores.max())  if len(neg_scores) else None,
        },
        'ks_statistic': float(ks_stat),
        'ks_p_value':   float(ks_p),
    }

    metrics = {
        # Core
        'accuracy':                  accuracy,
        'confusion_matrix':          conf_matrix.tolist(),
        'classification_report_str': class_report,
        # Ranking
        'roc_auc':                   roc_auc,
        'auprc':                     auprc,
        'roc_curve': {
            'fpr':        fpr.tolist(),
            'tpr':        tpr.tolist(),
            'thresholds': roc_thresholds.tolist(),
        },
        'precision_recall_curve': {
            'precision':  precision_curve.tolist(),
            'recall':     recall_curve.tolist(),
            'thresholds': pr_thresholds.tolist(),
        },
        # Calibration
        'brier_score': brier,
        'log_loss':    logloss,
        # Optimal threshold
        'optimal_threshold': {
            'value':     optimal_threshold,
            'f1':        f1_optimal,
            'precision': precision_optimal,
            'recall':    recall_optimal,
        },
        # Agreement
        'mcc':         mcc,
        'cohen_kappa': kappa,
        # Score distributions
        'score_distribution': score_distribution,
        # Legacy key kept for backwards compat with existing reports/MLflow imports
        'average_precision': auprc,
    }

    # ── Bootstrap confidence intervals (1000 resamples, 95% CI) ─────────────
    print("Computing bootstrap confidence intervals (1000 resamples)...")
    ci_results = _bootstrap_metric_cis(binary_true, probs_flat, predictions)
    metrics['confidence_intervals'] = ci_results

    # ── MC Dropout prediction intervals (50 stochastic passes) ───────────────
    print("Computing MC Dropout prediction intervals (50 passes)...")
    mc_results = _mc_dropout_prediction_intervals(model, test_loader, device)
    metrics['mc_dropout'] = mc_results

    # Update report with metrics
    report.update_metrics(metrics)

    # Save the test report
    report_path = report.save_report()
    print(f"\nTest report saved to: {report_path}")

    return metrics, report

if __name__ == '__main__':
    import argparse

    VALID_MODES = ["model-vs-model", "reference-vs-generated", "context-vs-generated"]

    parser = argparse.ArgumentParser(description="Evaluate a SilverBullet checkpoint on test data.")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=None,
        help=(
            "Evaluation mode to test. If given, loads data from data/{mode}/test.json and "
            "checkpoint from models/{mode}.pth. "
            "If omitted, loads from data/test.json and best_model.pth (general model)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit path to checkpoint file (overrides --mode default).",
    )
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs('cache', exist_ok=True)
    os.makedirs('test_reports', exist_ok=True)

    if args.checkpoint:
        ckpt_path = args.checkpoint
        data_dir  = f'data/{args.mode}' if args.mode else 'data'
    elif args.mode:
        from pathlib import Path as _Path
        new_ckpt  = str(_Path('models') / args.mode / 'best.pth')
        legacy    = f'models/{args.mode}.pth'
        ckpt_path = new_ckpt if os.path.exists(new_ckpt) else legacy
        data_dir  = f'data/{args.mode}'
        print(f"Mode: {args.mode}  |  Data: {data_dir}/  |  Checkpoint: {ckpt_path}")
    else:
        ckpt_path = 'best_model.pth'
        data_dir  = 'data'
        print("Mode: general  |  Data: data/  |  Checkpoint: best_model.pth")

    # Load test data
    test_pairs, test_labels = load_json_data(f'{data_dir}/test.json')
    print(f"Loaded {len(test_pairs)} test pairs")

    # Create dataset with feature caching enabled
    test_dataset = TextSimilarityDataset(test_pairs, test_labels, use_cache=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Load the model (auto-detects legacy Conv1D vs current Conv2D architecture)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, arch = _load_model_from_checkpoint(checkpoint, device)
    print(f"  architecture : {arch}")

    # Test the model and generate report
    metrics, report = test_model(model, test_loader, test_pairs, device=str(device))

    # Print summary results
    ot = metrics['optimal_threshold']
    sd = metrics['score_distribution']
    print("\nTest Results:")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  AUPRC          : {metrics['auprc']:.4f}")
    print(f"  MCC            : {metrics['mcc']:.4f}")
    print(f"  Cohen's Kappa  : {metrics['cohen_kappa']:.4f}")
    print(f"  Brier Score    : {metrics['brier_score']:.4f}  (lower = better calibration)")
    print(f"  Log Loss       : {metrics['log_loss']:.4f}")
    print(f"  KS Statistic   : {sd['ks_statistic']:.4f}  (p={sd['ks_p_value']:.4f})")
    print(f"  Optimal thresh : {ot['value']:.4f}  ->  F1={ot['f1']:.4f}  P={ot['precision']:.4f}  R={ot['recall']:.4f}")
    print(f"  Score dist     : pos mean={sd['positive_class']['mean']:.3f}  neg mean={sd['negative_class']['mean']:.3f}")
    ci = metrics.get('confidence_intervals', {})
    if ci:
        print("\n95% Bootstrap Confidence Intervals (n=1000):")
        for k, v in ci.items():
            print(f"  {k:<18}: {v['point']:.4f}  [{v['lower']:.4f}, {v['upper']:.4f}]")
    mc = metrics.get('mc_dropout', {})
    if mc:
        print(f"\nMC Dropout ({mc['n_passes']} passes, 95% PI):")
        print(f"  Mean prob std (epistemic uncertainty) : {mc['mean_std']:.4f}")
        print(f"  Mean interval width                   : {mc['mean_interval_width']:.4f}")
        print(f"  High-uncertainty samples (std > 0.1)  : {mc['high_uncertainty_frac']:.1%}")
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    print("\nClassification Report:")
    print(metrics['classification_report_str'])

    # Log test metrics to MLflow
    report_path = os.path.join(report.output_dir, f"test_report_{report.timestamp}.json")
    _log_test_to_mlflow(metrics, args.mode, report_path)
