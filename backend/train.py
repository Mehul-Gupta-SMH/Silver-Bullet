import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
from datetime import datetime
from backend.model import TextSimilarityCNN
from backend.Splitter.sentence_splitter import split_txt
from backend.Features.Semantic.getSemanticWeights import SemanticWeights
from backend.Features.Lexical.getLexicalWeights import LexicalWeights
from backend.Features.NLI.getNLIweights import NLIWeights
from backend.Features.EntityGroups.getOverlap import EntityMatch
from backend.Features.LCS.getLCSweights import LCSWeights
from backend.feature_cache import FeatureCache
from backend.feature_registry import FEATURE_KEYS, build_manifest
from backend.training_report import TrainingReport


class _Tracker:
    """Optional MLflow + Prometheus pushgateway tracker.

    Both backends are opt-in and fail silently — training continues even if
    MLflow or the pushgateway is unreachable.

    Environment variables:
        MLFLOW_TRACKING_URI          default: http://localhost:5000
        PROMETHEUS_PUSHGATEWAY_URL   default: http://localhost:9091
    """

    def __init__(self, mode: str, params: dict):
        self._mlflow = None
        self._push_url = None
        self._gauges = {}
        self._registry = None
        self._mode = mode or "general"
        self._run_name = f"{self._mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._params = params
        self._history: list[dict] = []   # buffers every epoch regardless of MLflow state
        self._run_id: str | None = None

        # --- MLflow ---
        try:
            import mlflow
            uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self._mode)
            mlflow.start_run(run_name=self._run_name)
            self._run_id = mlflow.active_run().info.run_id
            mlflow.log_params(params)
            self._mlflow = mlflow
            print(f"[MLflow] tracking at {uri}  run={self._run_name}  id={self._run_id}")
        except Exception as exc:
            print(f"[MLflow] not available ({exc}) — will retry import at finish.")

        # --- Prometheus pushgateway ---
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
            self._push_url = os.environ.get("PROMETHEUS_PUSHGATEWAY_URL", "http://localhost:9091")
            self._registry = CollectorRegistry()
            for name in ("train_loss", "val_loss", "accuracy", "epoch"):
                self._gauges[name] = Gauge(
                    f"silverbullet_{name}", f"SilverBullet training {name}",
                    ["mode", "run"], registry=self._registry,
                )
            self._push_fn = push_to_gateway
            print(f"[Prometheus] pushgateway at {self._push_url}")
        except Exception as exc:
            print(f"[Prometheus] not available ({exc}) — skipping.")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, accuracy: float):
        self._history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "accuracy": accuracy}
        )
        if self._mlflow:
            try:
                self._mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss, "accuracy": accuracy},
                    step=epoch,
                )
            except Exception:
                pass

        if self._registry:
            try:
                labels = {"mode": self._mode, "run": self._run_name}
                self._gauges["epoch"].labels(**labels).set(epoch)
                self._gauges["train_loss"].labels(**labels).set(train_loss)
                self._gauges["val_loss"].labels(**labels).set(val_loss)
                self._gauges["accuracy"].labels(**labels).set(accuracy)
                self._push_fn(self._push_url, job="silverbullet_training",
                              registry=self._registry)
            except Exception:
                pass

    def log_artifact(self, path: str):
        if self._mlflow:
            try:
                self._mlflow.log_artifact(path)
            except Exception:
                pass

    def finish(self, best_val_loss: float, best_epoch: int, model=None):
        # If MLflow wasn't reachable at startup, try once more now that training is done.
        if self._mlflow is None:
            try:
                import mlflow
                uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(self._mode)
                mlflow.start_run(run_name=self._run_name)
                self._run_id = mlflow.active_run().info.run_id
                mlflow.log_params(self._params)
                # Replay all buffered epoch metrics
                for row in self._history:
                    mlflow.log_metrics(
                        {"train_loss": row["train_loss"],
                         "val_loss":   row["val_loss"],
                         "accuracy":   row["accuracy"]},
                        step=row["epoch"],
                    )
                self._mlflow = mlflow
                print(f"[MLflow] late-connect succeeded — replayed {len(self._history)} epochs.")
            except Exception as exc:
                print(f"[MLflow] finish import failed ({exc}) — no experiment recorded.")

        if self._mlflow:
            try:
                self._mlflow.log_metrics(
                    {"best_val_loss": best_val_loss, "best_epoch": best_epoch}
                )
                # --- Model Registry ---
                if model is not None and self._run_id:
                    try:
                        self._mlflow.pytorch.log_model(model, "model")
                        model_name = f"silverbullet-{self._mode}"
                        self._mlflow.register_model(
                            f"runs:/{self._run_id}/model", model_name
                        )
                        print(f"[MLflow] model registered as '{model_name}'")
                    except Exception as reg_exc:
                        print(f"[MLflow] model registration failed ({reg_exc}) — skipped.")
                self._mlflow.end_run()
            except Exception:
                pass


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
                patience=5, min_delta=0.001, best_ckpt='best_model.pth', mode='general'):
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
        'mode':                      mode,
    }
    tracker = _Tracker(mode, training_params)
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
        tracker.log_epoch(epoch + 1, avg_train_loss, avg_val_loss, accuracy)

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
            }, best_ckpt)
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

    tracker.log_artifact(final_model_path)
    tracker.log_artifact(best_model_path)

    # Restore best weights so the registered model reflects the best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    tracker.finish(best_val_loss=best_val_loss, best_epoch=best_epoch + 1, model=model)

    print(f"Final model  -> {final_model_path}")
    print(f"Best model   -> {best_model_path}")
    print(f"Report       -> {report.save_report(intermediate=False)}")

    return model, report.report_data['training_metrics']


if __name__ == '__main__':
    import argparse

    VALID_MODES = ["model-vs-model", "reference-vs-generated", "context-vs-generated"]

    parser = argparse.ArgumentParser(description="Train a SilverBullet evaluation model.")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=None,
        help=(
            "Evaluation mode to train. If given, loads data from data/{mode}/ and saves "
            "checkpoint to models/{mode}.pth. "
            "If omitted, loads from data/ and saves to best_model.pth (general model)."
        ),
    )
    args = parser.parse_args()

    os.makedirs('cache', exist_ok=True)
    os.makedirs('training_reports', exist_ok=True)

    if args.mode:
        data_dir       = f'data/{args.mode}'
        checkpoint_dir = 'models'
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_ckpt = f'{checkpoint_dir}/{args.mode}.pth'
        print(f"Mode: {args.mode}  |  Data: {data_dir}/  |  Checkpoint: {best_ckpt}")
    else:
        data_dir  = 'data'
        best_ckpt = 'best_model.pth'
        print("Mode: general  |  Data: data/  |  Checkpoint: best_model.pth")

    train_pairs, train_labels = load_json_data(f'{data_dir}/train.json')
    val_pairs,   val_labels   = load_json_data(f'{data_dir}/validate.json')
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    train_dataset = TextSimilarityDataset(train_pairs, train_labels, use_cache=True)
    val_dataset   = TextSimilarityDataset(val_pairs,   val_labels,   use_cache=True)

    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=16)

    model = TextSimilarityCNN(num_features=train_dataset.num_features)
    trained_model, metrics = train_model(
        model, train_loader, val_loader, best_ckpt=best_ckpt,
        mode=args.mode or "general",
    )
    print("Training complete.")
