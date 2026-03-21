import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
from model import TextSimilarityCNN
from Splitter.sentence_splitter import split_txt
from Features.Semantic.getSemanticWeights import SemanticWeights
from Features.Lexical.getLexicalWeights import LexicalWeights
from Features.NLI.getNLIweights import NLIWeights
from Features.EntityGroups.getOverlap import EntityMatch
from Features.LCS.getLCSweights import LCSWeights
from feature_cache import FeatureCache
from feature_registry import FEATURE_KEYS, build_manifest
from training_report import TrainingReport


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs  = [[item['text1'], item['text2']] for item in data['data']]
    labels = [item['label'] for item in data['data']]
    return pairs, labels


def feature_map_to_tensor(feature_map: dict) -> torch.Tensor:
    """Stack all feature maps into a single [F, 64, 64] tensor.

    Maps are stacked in the canonical order defined by FEATURE_KEYS so that
    the channel index is always stable regardless of dict insertion order.

    Raises:
        KeyError: If feature_map contains unknown keys or is missing expected keys.

    Returns:
        torch.Tensor: shape [num_features, 64, 64]
    """
    unknown = set(feature_map) - set(FEATURE_KEYS)
    if unknown:
        raise KeyError(f"feature_map contains keys not in FEATURE_KEYS: {sorted(unknown)}")
    missing = set(FEATURE_KEYS) - set(feature_map)
    if missing:
        raise KeyError(f"feature_map is missing expected keys: {sorted(missing)}")

    maps = []
    for key in FEATURE_KEYS:
        val = feature_map[key]
        if isinstance(val, torch.Tensor):
            maps.append(val.float())
        else:
            maps.append(torch.tensor(val, dtype=torch.float32))
    return torch.stack(maps, dim=0)  # [F, 64, 64]


class TextSimilarityDataset(Dataset):
    """Extract features for every text pair and store as [F, 64, 64] tensors."""

    def __init__(self, paragraph_pairs, labels, use_cache=True):
        global FEATURE_ORDER
        self.labels = labels
        self.cache  = FeatureCache() if use_cache else None

        self.lexical  = LexicalWeights()
        self.semantic = SemanticWeights()
        self.nli      = NLIWeights()
        self.entity   = EntityMatch()
        self.lcs      = LCSWeights()

        tensors = []
        for pair in tqdm(paragraph_pairs, desc="Extracting features"):
            # ---- cache lookup ----
            if use_cache:
                cached = self.cache.get_features(pair[0], pair[1])
                if cached is not None:
                    tensors.append(torch.tensor(cached, dtype=torch.float32))
                    continue

            # ---- compute ----
            sent_group1 = split_txt(pair[0])
            sent_group2 = split_txt(pair[1])

            feature_map = {}
            feature_map.update(self.lexical.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.semantic.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.nli.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.entity.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.lcs.getFeatureMap(sent_group1, sent_group2))

            stacked = feature_map_to_tensor(feature_map)  # [F, 64, 64]

            # ---- cache save ----
            if use_cache:
                self.cache.save_features(pair[0], pair[1], stacked.tolist())

            tensors.append(stacked)

        # [N, F, 64, 64]
        self.features = torch.stack(tensors, dim=0)
        self.labels   = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

        print(f"Feature tensor shape : {self.features.shape}")   # [N, F, 64, 64]
        print(f"Labels shape         : {self.labels.shape}")

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001,
                patience=5, min_delta=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_features = train_loader.dataset.num_features

    training_params = {
        'num_epochs':                num_epochs,
        'learning_rate':             learning_rate,
        'batch_size':                train_loader.batch_size,
        'optimizer':                 'Adam',
        'loss_function':             'MSELoss',
        'early_stopping_patience':   patience,
        'early_stopping_min_delta':  min_delta,
        'num_feature_channels':      num_features,
    }
    report = TrainingReport(model, training_params)
    report.update_dataset_info(
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        feature_dim=num_features,
    )

    model     = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss      = float('inf')
    patience_counter   = 0
    best_model_state   = None
    best_optimizer_state = None
    best_epoch         = 0

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct  = 0
        total    = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs   = model(features)
                val_loss += criterion(outputs, labels).item()
                predicted  = (outputs > 0.5).float()
                total     += labels.size(0)
                correct   += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy     = 100 * correct / total

        report.update_epoch_metrics(epoch + 1, avg_train_loss, avg_val_loss, accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]  '
              f'Train: {avg_train_loss:.4f}  Val: {avg_val_loss:.4f}  Acc: {accuracy:.2f}%')

        # --- Early stopping / checkpoint ---
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss        = avg_val_loss
            patience_counter     = 0
            best_epoch           = epoch
            best_model_state     = model.state_dict()
            best_optimizer_state = optimizer.state_dict()

            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                 avg_val_loss,
                'accuracy':             accuracy,
                'num_features':         num_features,
                'manifest':             build_manifest(),
            }, 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs "
                  f"(best val loss {best_val_loss:.4f} @ epoch {best_epoch + 1})")
            break

    # --- Save final artefacts ---
    ts                = report.timestamp
    final_model_path  = f'model_weights_{ts}_final.pth'
    best_model_path   = f'model_weights_{ts}_best.pth'

    manifest = build_manifest()

    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss':           avg_val_loss,
        'best_loss':            best_val_loss,
        'final_accuracy':       accuracy,
        'num_features':         num_features,
        'manifest':             manifest,
    }, final_model_path)

    torch.save({
        'epoch':                best_epoch,
        'model_state_dict':     best_model_state,
        'optimizer_state_dict': best_optimizer_state,
        'loss':                 best_val_loss,
        'num_features':         num_features,
        'manifest':             manifest,
    }, best_model_path)

    print(f"Final model  -> {final_model_path}")
    print(f"Best model   -> {best_model_path}")
    print(f"Report       -> {report.save_report(intermediate=False)}")

    return model, report.report_data['training_metrics']


if __name__ == '__main__':
    os.makedirs('cache', exist_ok=True)
    os.makedirs('training_reports', exist_ok=True)

    train_pairs, train_labels = load_json_data('data/train.json')
    val_pairs,   val_labels   = load_json_data('data/validate.json')
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    train_dataset = TextSimilarityDataset(train_pairs, train_labels, use_cache=True)
    val_dataset   = TextSimilarityDataset(val_pairs,   val_labels,   use_cache=True)

    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=16)

    model = TextSimilarityCNN(num_features=train_dataset.num_features)
    trained_model, metrics = train_model(model, train_loader, val_loader)
    print("Training complete.")
