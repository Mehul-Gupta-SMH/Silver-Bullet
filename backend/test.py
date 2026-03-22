import torch
from backend.model import TextSimilarityCNN, TextSimilarityCNNLegacy
from backend.predict import _load_model_from_checkpoint
from backend.train import TextSimilarityDataset, load_json_data
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
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
        metrics = self.report_data['metrics']
        md_content += f"- Accuracy: {metrics['accuracy']:.4f}\n"
        md_content += f"- ROC AUC: {metrics['roc_auc']:.4f}\n"
        md_content += f"- Average Precision: {metrics['average_precision']:.4f}\n"

        md_content += "\n### Classification Report\n```\n"
        md_content += metrics['classification_report_str']
        md_content += "\n```\n"

        md_content += "\n### Confusion Matrix\n```\n"
        md_content += str(np.array(metrics['confusion_matrix']))
        md_content += "\n```\n"

        md_path = Path(json_report_path).with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

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
        for i, (features, labels) in enumerate(test_loader):
            features, labels = features.to(device), labels.to(device)
            if is_legacy:
                trained_num_maps = model.fc_reduce1.in_features // (64 * 64)
                features = features[:, :trained_num_maps, :, :]
                features = features.view(features.size(0), -1)
            outputs = model(features)
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

    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)

    # Binarise true labels at 0.5 so sklearn metrics work with both binary (0/1)
    # and continuous float labels (e.g. 0.7 faithfulness scores).
    binary_true = (true_labels >= 0.5).astype(int)

    # Calculate metrics
    conf_matrix = confusion_matrix(binary_true, predictions)
    class_report = classification_report(binary_true, predictions, target_names=['Not Similar', 'Similar'])
    roc_auc = roc_auc_score(binary_true, probabilities)
    precision, recall, _ = precision_recall_curve(binary_true, probabilities)
    avg_precision = np.mean(precision)

    metrics = {
        'accuracy': (predictions == binary_true).mean(),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report_str': class_report,
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'precision_recall_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
    }

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
        ckpt_path = f'models/{args.mode}.pth'
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
    print("\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    print("\nClassification Report:")
    print(metrics['classification_report_str'])
