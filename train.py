import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
import os
from model import TextSimilarityCNN
from Splitter.sentence_splitter import split_txt
from Features.Semantic.getSemanticWeights import SemanticWeights
from Features.Lexical.getLexicalWeights import LexicalWeights
from Features.NLI.getNLIweights import NLIWeights
from Features.EntityGroups.getOverlap import EntityMatch
from feature_cache import FeatureCache
from training_report import TrainingReport

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs = [[item['text1'], item['text2']] for item in data['data']]
    labels = [item['label'] for item in data['data']]
    return pairs, labels

class TextSimilarityDataset(Dataset):
    def __init__(self, paragraph_pairs, labels, use_cache=True):
        self.features = []
        self.labels = labels
        self.cache = FeatureCache() if use_cache else None

        # Initialize feature extractors
        self.lexical = LexicalWeights()
        self.semantic = SemanticWeights()
        self.nli = NLIWeights()
        self.entity = EntityMatch()

        # Process all pairs and extract features
        for pair in tqdm(paragraph_pairs, desc="Extracting features"):
            if use_cache:
                # Try to get features from cache
                cached_features = self.cache.get_features(pair[0], pair[1])
                if cached_features is not None:
                    self.features.append(cached_features)
                    continue

            # If not in cache, compute features
            sent_group1 = split_txt(pair[0])
            sent_group2 = split_txt(pair[1])

            feature_map = {}
            feature_map.update(self.lexical.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.semantic.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.nli.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.entity.getFeatureMap(sent_group1, sent_group2))

            # Convert feature map to vector
            feature_vector = []
            for value in feature_map.values():
                if isinstance(value, (list, np.ndarray)):
                    feature_vector.extend([float(x) for x in np.array(value).flatten()])
                elif isinstance(value, torch.Tensor):
                    feature_vector.extend([float(x) for x in value.detach().cpu().numpy().flatten()])
                else:
                    try:
                        feature_vector.append(float(np.array(value).item()))
                    except:
                        if isinstance(value, (int, float)):
                            feature_vector.append(float(value))
                        else:
                            print(f"Warning: Skipping invalid feature value: {value}")

            # Cache the computed features
            if use_cache:
                self.cache.save_features(pair[0], pair[1], feature_vector)

            self.features.append(feature_vector)

        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

        print(f"Feature vector size: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001,
             patience=5, min_delta=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize training report
    training_params = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': train_loader.batch_size,
        'optimizer': 'Adam',
        'loss_function': 'BCELoss',
        'early_stopping_patience': patience,
        'early_stopping_min_delta': min_delta
    }
    report = TrainingReport(model, training_params)

    # Update dataset information
    report.update_dataset_info(
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        feature_dim=train_loader.dataset.features.shape[1]
    )

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_optimizer_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        # Update training report for this epoch
        report.update_epoch_metrics(epoch + 1, avg_train_loss, avg_val_loss, accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {accuracy:.2f}%')

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:  # Improvement
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch

            # Save best model state
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'accuracy': accuracy
            }, 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            print(f"Best validation loss ({best_val_loss:.4f}) was achieved at epoch {best_epoch + 1}")
            break

    # Save final model and best model weights separately
    final_model_path = f'model_weights_{report.timestamp}_final.pth'
    best_model_path = f'model_weights_{report.timestamp}_best.pth'

    # Save final weights
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_val_loss,
        'best_loss': best_val_loss,
        'final_accuracy': accuracy
    }, final_model_path)

    # Save best weights
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': best_optimizer_state,
        'loss': best_val_loss
    }, best_model_path)

    print(f"\nFinal model weights saved to: {final_model_path}")
    print(f"Best model weights saved to: {best_model_path}")

    # Save final report
    final_report_path = report.save_report(intermediate=False)
    print(f"Final training report saved to: {final_report_path}")

    return model, report.report_data['training_metrics']

if __name__ == '__main__':
    # Create cache directory if it doesn't exist
    os.makedirs('cache', exist_ok=True)
    os.makedirs('training_reports', exist_ok=True)

    # Load data from JSON files
    train_pairs, train_labels = load_json_data('data/train.json')
    val_pairs, val_labels = load_json_data('data/validate.json')

    print(f"Loaded {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")

    # Create datasets with feature caching enabled
    train_dataset = TextSimilarityDataset(train_pairs, train_labels, use_cache=True)
    val_dataset = TextSimilarityDataset(val_pairs, val_labels, use_cache=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize and train model
    input_dim = len(train_dataset[0][0])
    model = TextSimilarityCNN(input_dim=input_dim)

    trained_model, metrics = train_model(model, train_loader, val_loader)
    print("Training completed! Model and training report saved.")
