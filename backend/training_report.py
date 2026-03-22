import json
import os
from datetime import datetime
from typing import Dict, Any
import torch
from pathlib import Path

class TrainingReport:
    def __init__(self, model, training_params: Dict[str, Any], output_dir='training_reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'model_architecture': self._get_model_architecture(model),
            'training_parameters': training_params,
            'training_metrics': {
                'epochs': [],
                'train_losses': [],
                'val_losses': [],
                'val_accuracies': [],
                'best_epoch': 0,
                'best_val_loss': float('inf'),
                'best_val_accuracy': 0.0
            },
            'dataset_info': {},
            'hardware_info': self._get_hardware_info()
        }

    def _get_model_architecture(self, model):
        """Extract model architecture details"""
        return {
            'name': model.__class__.__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'layer_info': str(model)
        }

    def _get_hardware_info(self):
        """Get hardware information"""
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }

    def update_dataset_info(self, train_size: int, val_size: int, feature_dim: int):
        """Update dataset information"""
        self.report_data['dataset_info'] = {
            'training_samples': train_size,
            'validation_samples': val_size,
            'feature_dimension': feature_dim,
            'cached_features_used': True  # Assuming we're using feature caching
        }

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update training metrics"""
        self.report_data['training_metrics'] = metrics

    def add_validation_results(self, results: Dict[str, Any]):
        """Add validation results"""
        self.report_data['validation_results'] = results

    def update_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float, val_accuracy: float):
        """Update metrics for current epoch"""
        self.report_data['training_metrics']['epochs'].append(epoch)
        self.report_data['training_metrics']['train_losses'].append(float(train_loss))
        self.report_data['training_metrics']['val_losses'].append(float(val_loss))
        self.report_data['training_metrics']['val_accuracies'].append(float(val_accuracy))

        # Update best metrics
        if val_loss < self.report_data['training_metrics']['best_val_loss']:
            self.report_data['training_metrics']['best_val_loss'] = float(val_loss)
            self.report_data['training_metrics']['best_val_accuracy'] = float(val_accuracy)
            self.report_data['training_metrics']['best_epoch'] = epoch

        # Save intermediate report
        self.save_report(intermediate=True)

    def save_report(self, intermediate=False):
        """Save the training report"""
        if intermediate:
            report_file = os.path.join(self.output_dir, f'training_report_{self.timestamp}_current.json')
        else:
            report_file = os.path.join(self.output_dir, f'training_report_{self.timestamp}_final.json')

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2)

        # Generate markdown summary
        self._generate_markdown_summary(report_file)

        return report_file

    def _generate_markdown_summary(self, json_report_path):
        """Generate a markdown summary of the training report"""
        md_content = f"""# Training Report Summary
Generated on: {self.report_data['timestamp']}

## Model Architecture
- Model Name: {self.report_data['model_architecture']['name']}
- Total Parameters: {self.report_data['model_architecture']['total_parameters']:,}
- Trainable Parameters: {self.report_data['model_architecture']['trainable_parameters']:,}

## Training Parameters
"""
        for k, v in self.report_data['training_parameters'].items():
            md_content += f"- {k}: {v}\n"

        md_content += f"""
## Dataset Information
- Training Samples: {self.report_data['dataset_info']['training_samples']}
- Validation Samples: {self.report_data['dataset_info']['validation_samples']}
- Feature Dimension: {self.report_data['dataset_info']['feature_dimension']}
- Using Cached Features: {self.report_data['dataset_info']['cached_features_used']}

## Hardware Information
- Device: {self.report_data['hardware_info']['device']}
"""
        if self.report_data['hardware_info']['cuda_device']:
            md_content += f"- GPU: {self.report_data['hardware_info']['cuda_device']}\n"

        md_content += """
## Training Metrics
"""
        if 'training_metrics' in self.report_data:
            metrics = self.report_data['training_metrics']
            md_content += f"- Current Epoch: {len(metrics['epochs'])}\n"
            md_content += f"- Best Epoch: {metrics['best_epoch']}\n"
            md_content += f"- Best Validation Loss: {metrics['best_val_loss']:.4f}\n"
            md_content += f"- Best Validation Accuracy: {metrics['best_val_accuracy']:.2f}%\n"

            if len(metrics['epochs']) > 0:
                md_content += f"- Latest Train Loss: {metrics['train_losses'][-1]:.4f}\n"
                md_content += f"- Latest Validation Loss: {metrics['val_losses'][-1]:.4f}\n"
                md_content += f"- Latest Validation Accuracy: {metrics['val_accuracies'][-1]:.2f}%\n"

        md_path = Path(json_report_path).with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
