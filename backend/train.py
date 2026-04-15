import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import subprocess
import time
import urllib.request
import urllib.error

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
from datetime import datetime
from pathlib import Path
from backend.model import TextSimilarityCNN
from backend.Splitter.sentence_splitter import split_txt
from backend.Features.Semantic.getSemanticWeights import SemanticWeights
from backend.Features.Lexical.getLexicalWeights import LexicalWeights
from backend.Features.NLI.getNLIweights import NLIWeights
from backend.Features.EntityGroups.getOverlap import EntityMatch
from backend.Features.LCS.getLCSweights import LCSWeights
from backend.Features.Numeric.getNumericGrounding import NumericGrounding
from backend.Features.Relations.getRelationWeights import RelationGrounding
from backend.Features.Relations.getRelexWeights import RelexGrounding
from backend.feature_cache import FeatureCache
from backend.feature_registry import FEATURE_KEYS, SPATIAL_SIZE, build_manifest, get_feature_keys
from backend.training_report import TrainingReport


_MLFLOW_SERVER_PROC: subprocess.Popen | None = None  # module-level handle so the process outlives the function


def _ensure_mlflow_server(uri: str, timeout: int = 30) -> None:
    """Start the MLflow tracking server if it is not already reachable.

    Probes *uri* with a lightweight HTTP request.  If the server is down,
    launches ``mlflow server`` in the background using the paths defined in
    CLAUDE.md (sqlite backend + ./mlflow/artifacts).  Waits up to *timeout*
    seconds for the server to accept connections before giving up.

    The spawned process is kept alive for the duration of the Python session.
    """
    global _MLFLOW_SERVER_PROC

    health_url = uri.rstrip("/") + "/health"

    def _reachable() -> bool:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                return resp.status < 500
        except Exception:
            return False

    if _reachable():
        return  # already up

    print(f"[MLflow] server not reachable at {uri} — starting it now…")

    mlflow_dir = Path("mlflow")
    mlflow_dir.mkdir(exist_ok=True)
    (mlflow_dir / "artifacts").mkdir(exist_ok=True)

    cmd = [
        "mlflow", "server",
        "--host", "127.0.0.1",
        "--port", "5000",
        "--backend-store-uri", f"sqlite:///{mlflow_dir / 'mlflow.db'}",
        "--default-artifact-root", str(mlflow_dir / "artifacts"),
    ]

    _MLFLOW_SERVER_PROC = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _reachable():
            print(f"[MLflow] server started (pid={_MLFLOW_SERVER_PROC.pid})  tracking at {uri}")
            return
        time.sleep(1)

    print(f"[MLflow] WARNING: server did not become reachable within {timeout}s — continuing without it.")


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
            _ensure_mlflow_server(uri)
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


def feature_map_to_tensor(feature_map: dict, feature_keys: list[str] | None = None) -> torch.Tensor:
    """Stack feature maps into a single [F, S, S] tensor.

    Maps are stacked in the canonical order defined by *feature_keys* (defaults to
    the global FEATURE_KEYS) so that the channel index is always stable regardless of
    dict insertion order.  Extra keys in feature_map are silently ignored — only the
    requested subset is selected.

    Args:
        feature_map: Dict mapping feature key → 2-D array/tensor (SPATIAL_SIZE×SPATIAL_SIZE).
        feature_keys: Ordered list of keys to stack. Defaults to the global FEATURE_KEYS.

    Raises:
        KeyError: If feature_map is missing any key listed in feature_keys.

    Returns:
        torch.Tensor: shape [len(feature_keys), SPATIAL_SIZE, SPATIAL_SIZE]
    """
    keys = feature_keys if feature_keys is not None else FEATURE_KEYS
    missing = set(keys) - set(feature_map)
    if missing:
        raise KeyError(f"feature_map is missing expected keys: {sorted(missing)}")

    maps = []
    for key in keys:
        val = feature_map[key]
        if isinstance(val, torch.Tensor):
            maps.append(val.float())
        else:
            maps.append(torch.tensor(val, dtype=torch.float32))
    return torch.stack(maps, dim=0)  # [F, S, S]



def _cache_entry_complete(entry, feature_keys: list[str] | None = None) -> bool:
    """Return True if a cache entry (dict format) has all keys in *feature_keys*."""
    if not isinstance(entry, dict):
        return False
    keys = feature_keys if feature_keys is not None else FEATURE_KEYS
    return all(k in entry for k in keys)


def _missing_groups(entry: dict, feature_keys: list[str] | None = None) -> set[str]:
    """Return the set of extractor group names whose keys are absent from *entry*.

    Groups: 'lexical', 'semantic', 'nli', 'entity', 'lcs', 'numeric'.
    Only groups that produce at least one key listed in *feature_keys* that is
    missing from *entry* are returned.
    """
    keys = feature_keys if feature_keys is not None else FEATURE_KEYS
    missing_keys = [k for k in keys if k not in entry]
    groups: set[str] = set()
    for k in missing_keys:
        if k.startswith(("PREC_", "REC_", "SOFT_")) or "/" in k:
            groups.add("semantic")
        elif k in ("entailment", "neutral", "contradiction"):
            groups.add("nli")
        elif k == "numeric_jaccard":
            groups.add("numeric")
        elif k in ("entity_grounding_recall", "relation_triplet_recall"):
            groups.add("relations")
        elif k.startswith("entity_"):
            groups.add("entity")
        elif k in ("lcs_token", "lcs_char"):
            groups.add("lcs")
        else:
            groups.add("lexical")
    return groups


def _prefill_semantic_cache(paragraph_pairs, cache: "FeatureCache | None",
                            feature_keys: list[str] | None = None) -> None:
    """Pre-encode all unique sentences from uncached pairs in one large batch per model.

    For each sentence-transformer model, this issues a single model.encode() call
    covering every unseen sentence across all pairs that will need feature extraction.
    The embeddings are stored in SemanticFeatures._embedding_cache so that the
    per-pair getFeatureMap() calls find them already computed and skip encoding entirely.

    Pairs that are fully covered by the feature cache are excluded — their sentences
    never need to be split or encoded.
    """
    from backend.Features.Semantic.__generate_semantic_features import SemanticFeatures
    import torch

    # Collect sentences only from pairs that need semantic encoding:
    # either fully uncached, or cached but missing semantic keys.
    unique_sentences: set[str] = set()
    for pair in paragraph_pairs:
        if cache is not None:
            entry = cache.get_features(pair[0], pair[1])
            if entry is not None and "semantic" not in _missing_groups(entry, feature_keys) if isinstance(entry, dict) else False:
                continue
        for sent in split_txt(pair[0]) + split_txt(pair[1]):
            unique_sentences.add(sent)

    if not unique_sentences:
        return

    sentences = list(unique_sentences)
    print(f"[SemanticCache] pre-encoding {len(sentences)} unique sentences across all uncached pairs…")

    # Reuse the shared model list from SemanticFeatures.
    # Only bulk-prefill the FIRST model (mxbai). Subsequent models (Qwen) are
    # skipped here and handled lazily per-pair during training. Bulk-encoding
    # Qwen after mxbai has already filled the page file causes a hard OOM crash
    # (exit 139) because the 32-sentence batches × 28 attention layers require
    # hundreds of MB per batch on top of the 2.4 GB model weights.
    import gc
    dummy = SemanticFeatures()
    models_to_prefill = dummy.feature_model_local_list[:1]
    for model_meta in tqdm(models_to_prefill, desc="Pre-encoding models"):
        model_name = model_meta["model"]
        # Load SQLite cache BEFORE checking new_sentences — otherwise all sentences
        # appear new because the in-memory dict is empty at process start.
        SemanticFeatures.load_embedding_cache(model_name)
        sent_cache = SemanticFeatures._embedding_cache.setdefault(model_name, {})
        new_sentences = [s for s in sentences if s not in sent_cache]
        if not new_sentences:
            print(f"[SemanticCache] all {len(sentences)} sentences already cached for {model_name} — skipping encode.")
            continue

        try:
            model = dummy.__load_model__(model_meta)
            embeddings = model.encode(
                new_sentences,
                task=model_meta["task"],
                prompt=model_meta["prompt"],
                show_progress_bar=True,
            )
            for s, emb in zip(new_sentences, embeddings):
                sent_cache[s] = emb

            # Persist new embeddings to SQLite now — the per-pair loop will find
            # them in-memory and skip encoding, so save_embedding_cache() won't be
            # called again for these sentences during the training loop.
            SemanticFeatures.save_embedding_cache(model_name, new_sentences)

            # Explicitly unload after encoding to free RAM before the next model loads.
            # Keeps embeddings in sent_cache (numpy arrays) but drops the model weights.
            SemanticFeatures._model_cache.pop(model_name, None)
            del model
            gc.collect()

        except (OSError, MemoryError, Exception) as exc:
            safe = str(exc).encode("ascii", "replace").decode("ascii")
            print(f"[SemanticCache] WARNING: pre-encoding failed for '{model_name}' ({safe}). "
                  "Will use per-pair lazy encoding for this model.")
            # Remove any partially-loaded model from cache to avoid using corrupt state
            SemanticFeatures._model_cache.pop(model_name, None)
            gc.collect()

    print(f"[SemanticCache] done — embedding cache now covers "
          f"{sum(len(v) for v in SemanticFeatures._embedding_cache.values())} sentence-model entries.")


def _prefill_entity_cache(paragraph_pairs, cache: "FeatureCache | None",
                          feature_keys: list[str] | None = None) -> None:
    """Pre-run GLiNER NER on all unique sentences from uncached pairs in one batch.

    EntityMatch._entity_cache maps sentence → {entity_type: count}.  By filling
    it before the per-pair loop, every getFeatureMap() call finds its sentences
    already processed and skips the GLiNER model entirely.
    """
    from backend.Features.EntityGroups.getOverlap import EntityMatch

    # If entity features aren't in this mode's basket, skip entirely.
    keys = feature_keys if feature_keys is not None else FEATURE_KEYS
    if not any(k.startswith("entity_") for k in keys):
        return

    unique_sentences: set[str] = set()
    for pair in paragraph_pairs:
        if cache is not None:
            entry = cache.get_features(pair[0], pair[1])
            if entry is not None and "entity" not in _missing_groups(entry, feature_keys) if isinstance(entry, dict) else False:
                continue
        for sent in split_txt(pair[0]) + split_txt(pair[1]):
            unique_sentences.add(sent)

    # Load SQLite cache BEFORE filtering — avoids treating all sentences as new
    # when the in-memory dict is empty at process start.
    EntityMatch.load_entity_cache()
    new_sentences = [s for s in unique_sentences if s not in EntityMatch._entity_cache]
    if not new_sentences:
        print(f"[EntityCache] all {len(unique_sentences)} sentences already cached — skipping NER.")
        return

    print(f"[EntityCache] pre-running NER on {len(new_sentences)} unique sentences…")
    dummy = EntityMatch()
    # Process in chunks to avoid exhausting RAM on large/long-document datasets
    _CHUNK = 64
    for i in tqdm(range(0, len(new_sentences), _CHUNK), desc="Entity pre-encoding"):
        dummy._batch_get_entities(new_sentences[i : i + _CHUNK])
    print(f"[EntityCache] done — entity cache now covers {len(EntityMatch._entity_cache)} sentences.")


def _prefill_nli_cache(paragraph_pairs, cache: "FeatureCache | None",
                       feature_keys: list[str] | None = None) -> None:
    """Pre-score all unique sentence pairs with NLI in one large batched pass.

    NLIWeights._pair_cache maps (sent1, sent2) → (entailment, neutral, contradiction).
    By filling it before the per-pair training loop, every getFeatureMap() call finds
    its sentence pairs already scored and skips the roberta-large-mnli model entirely.

    Unlike semantic/entity pre-fills (per-sentence), NLI requires both sentences
    together (cross-encoder), so we cache sentence *pairs* from the full cartesian
    product of split_txt(text1) × split_txt(text2) for each training example.
    """
    from backend.Features.NLI.getNLIweights import NLIWeights

    keys = feature_keys if feature_keys is not None else FEATURE_KEYS
    if not any(k in ("entailment", "neutral", "contradiction") for k in keys):
        return

    # Load SQLite cache BEFORE building unique_pairs — otherwise every pair
    # appears uncached because the in-memory dict is empty at process start.
    NLIWeights.load_pair_cache()

    # Collect all unique sentence pairs from pairs that still need NLI computation.
    unique_pairs: set[tuple[str, str]] = set()
    for pair in paragraph_pairs:
        if cache is not None:
            entry = cache.get_features(pair[0], pair[1])
            if entry is not None and "nli" not in _missing_groups(entry, feature_keys) if isinstance(entry, dict) else False:
                continue
        sents1 = split_txt(pair[0])
        sents2 = split_txt(pair[1])
        for s1 in sents1:
            for s2 in sents2:
                if (s1, s2) not in NLIWeights._pair_cache:
                    unique_pairs.add((s1, s2))

    if not unique_pairs:
        print(f"[NLICache] all pairs already cached — skipping NLI prefill.")
        return

    all_pairs = list(unique_pairs)
    print(f"[NLICache] pre-scoring {len(all_pairs)} unique sentence pairs…")

    dummy = NLIWeights()
    dummy.phrase_list1 = [p[0] for p in all_pairs]
    dummy.phrase_list2 = [p[1] for p in all_pairs]
    # Run __calc_weights__ with a flat 1×N layout so all pairs go through the
    # batched inference path and land in _pair_cache.  We don't use the resulting
    # matrices — the cache population is the side-effect we want.
    # Re-use the existing batched inference logic by treating each pair as a 1×1 cell.
    if dummy.ModelName not in NLIWeights._model_cache:
        dummy.__load_model__()

    mc = NLIWeights._model_cache[dummy.ModelName]
    tok, mdl, device = mc["tokenizer"], mc["model"], mc["device"]
    id2label = mdl.config.id2label
    label_to_idx = {v.lower(): k for k, v in id2label.items()}

    import torch
    _BS = dummy.__batch_size__
    with torch.no_grad():
        for start in tqdm(range(0, len(all_pairs), _BS), desc="NLI pre-scoring"):
            chunk = all_pairs[start:start + _BS]
            enc = tok(
                [p[0] for p in chunk],
                [p[1] for p in chunk],
                padding=True, truncation=True,
                max_length=dummy.__max_len__,
                return_tensors="pt",
            ).to(device)
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_rows: list[tuple] = []
            for (s1, s2), row in zip(chunk, probs):
                scores = (
                    float(row[label_to_idx["entailment"]]),
                    float(row[label_to_idx["neutral"]]),
                    float(row[label_to_idx["contradiction"]]),
                )
                NLIWeights._pair_cache[(s1, s2)] = scores
                batch_rows.append((s1, s2, scores[0], scores[1], scores[2]))
            # Persist each batch immediately — crash-safe.
            NLIWeights.save_pair_cache(batch_rows)

    print(f"[NLICache] done — pair cache now covers {len(NLIWeights._pair_cache)} sentence pairs.")


def _prefill_relex_cache(paragraph_pairs, cache: "FeatureCache | None",
                         feature_keys: list[str] | None = None) -> None:
    """Pre-run RelexGrounding (gliner-relex) on all unique sentences before the per-pair loop.

    Without this, any pair missing 'relation_triplet_recall' in its cache entry
    will invoke the relex model sequentially inside the training loop — one pair
    at a time — rather than batching all new sentences in a single pass.

    Uses the same batched entity-extraction path as RelexGrounding._extract_triplets(),
    which issues one batch_predict_entities() call for all new sentences and then
    runs predict_relations() per sentence (unavoidable — it needs per-sentence entities).
    """
    from backend.Features.Relations.getRelexWeights import RelexGrounding

    keys = feature_keys if feature_keys is not None else FEATURE_KEYS
    if "relation_triplet_recall" not in keys:
        return

    unique_sentences: set[str] = set()
    for pair in paragraph_pairs:
        if cache is not None:
            entry = cache.get_features(pair[0], pair[1])
            if entry is not None and "relations" not in _missing_groups(entry, feature_keys) if isinstance(entry, dict) else False:
                continue
        for sent in split_txt(pair[0]) + split_txt(pair[1]):
            unique_sentences.add(sent)

    _MAX = RelexGrounding._MAX_CHARS
    # Load SQLite cache BEFORE filtering — otherwise all sentences appear new
    # because the in-memory dict is empty at process start.
    RelexGrounding.load_triplet_cache()
    new_sentences = [
        s for s in unique_sentences
        if s[:_MAX] not in RelexGrounding._triplet_cache
    ]
    if not new_sentences:
        print(f"[RelexCache] all {len(unique_sentences)} sentences already cached — skipping prefill.")
        return

    print(f"[RelexCache] pre-extracting triplets for {len(new_sentences)} unique sentences…")
    dummy = RelexGrounding()
    # Process in chunks so (a) tqdm shows chunk-level progress and
    # (b) results are persisted to SQLite after each chunk — crash-safe.
    # Chunk=32 (was 128) to stay within GPU/RAM budget — gliner-relex is ~3 GB peak.
    # gc.collect() after each chunk forces Python to release intermediate tensors so
    # PyTorch's CPU allocator doesn't accumulate fragmented blocks across chunks.
    import gc as _gc_relex
    _CHUNK = 32
    for i in tqdm(range(0, len(new_sentences), _CHUNK), desc="Relex prefill"):
        dummy._extract_triplets(new_sentences[i : i + _CHUNK])
        _gc_relex.collect()
    print(f"[RelexCache] done — triplet cache now covers {len(RelexGrounding._triplet_cache)} sentences.")


class TextSimilarityDataset(Dataset):
    """Extract features for every text pair and store as [F, S, S] tensors.

    Args:
        paragraph_pairs: List of [text1, text2] pairs.
        labels:          Corresponding float labels.
        use_cache:       Whether to read/write the feature cache.
        feature_keys:    Ordered list of feature keys to include in each tensor.
                         Defaults to the global FEATURE_KEYS. Pass a mode-specific
                         list (from get_feature_keys(mode)) for per-mode training.
    """

    def __init__(self, paragraph_pairs, labels, use_cache=True,
                 feature_keys: list[str] | None = None):
        self.labels       = labels
        self.cache        = FeatureCache() if use_cache else None
        self.feature_keys = feature_keys if feature_keys is not None else FEATURE_KEYS

        self.lexical   = LexicalWeights()
        self.semantic  = SemanticWeights()
        self.nli       = NLIWeights()
        self.entity    = EntityMatch()
        self.lcs       = LCSWeights()
        self.numeric   = NumericGrounding()
        self.relations = RelationGrounding()
        self.relex     = RelexGrounding()

        # Pre-encode / pre-run all sentence-level models in one large batch before
        # the per-pair loop so each getFeatureMap() call hits the in-memory cache only.
        _prefill_semantic_cache(paragraph_pairs, self.cache, self.feature_keys)
        _prefill_entity_cache(paragraph_pairs, self.cache, self.feature_keys)
        _prefill_nli_cache(paragraph_pairs, self.cache, self.feature_keys)
        _prefill_relex_cache(paragraph_pairs, self.cache, self.feature_keys)

        tensors = []
        for pair in tqdm(paragraph_pairs, desc="Extracting features"):
            # ---- cache lookup ----
            if use_cache:
                cached = self.cache.get_features(pair[0], pair[1])
                if cached is not None:
                    if isinstance(cached, dict):
                        if _cache_entry_complete(cached, self.feature_keys):
                            # Complete dict → reconstruct stacked tensor directly
                            fm = {k: torch.tensor(v, dtype=torch.float32) for k, v in cached.items()}
                            tensors.append(feature_map_to_tensor(fm, self.feature_keys))
                            continue
                        # Incomplete dict — run only missing extractor groups, merge, re-save
                        groups = _missing_groups(cached, self.feature_keys)
                        sent_group1 = split_txt(pair[0])
                        sent_group2 = split_txt(pair[1])
                        patch: dict = {}
                        if "lexical"   in groups: patch.update(self.lexical.getFeatureMap(sent_group1, sent_group2))
                        if "semantic"  in groups: patch.update(self.semantic.getFeatureMap(sent_group1, sent_group2))
                        if "nli"       in groups: patch.update(self.nli.getFeatureMap(sent_group1, sent_group2))
                        if "entity"    in groups: patch.update(self.entity.getFeatureMap(sent_group1, sent_group2))
                        if "lcs"       in groups: patch.update(self.lcs.getFeatureMap(sent_group1, sent_group2))
                        if "numeric"   in groups: patch.update(self.numeric.getFeatureMap(sent_group1, sent_group2))
                        if "relations" in groups:
                            patch.update(self.relations.getFeatureMap(sent_group1, sent_group2))
                            patch.update(self.relex.getFeatureMap(sent_group1, sent_group2))
                        merged = dict(cached)
                        merged.update({k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in patch.items()})
                        if use_cache:
                            self.cache.save_features(pair[0], pair[1], merged)
                        fm = {k: torch.tensor(v, dtype=torch.float32) for k, v in merged.items()}
                        tensors.append(feature_map_to_tensor(fm, self.feature_keys))
                        continue
                    else:
                        # Legacy format: stacked tensor as flat list — only use if
                        # feature count matches active feature set (guards against stale
                        # cache entries from a previous feature set version).
                        t = torch.tensor(cached, dtype=torch.float32)
                        if t.shape[0] == len(self.feature_keys):
                            tensors.append(t)
                            continue
                        # Shape mismatch — fall through to full recompute

            # ---- full compute (no usable cache entry) ----
            sent_group1 = split_txt(pair[0])
            sent_group2 = split_txt(pair[1])

            # Always compute all 5 extractor groups so the cache entry is complete
            # for all future modes (cache is mode-agnostic — stores all features).
            feature_map = {}
            feature_map.update(self.lexical.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.semantic.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.nli.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.entity.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.lcs.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.numeric.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.relations.getFeatureMap(sent_group1, sent_group2))
            feature_map.update(self.relex.getFeatureMap(sent_group1, sent_group2))

            stacked = feature_map_to_tensor(feature_map, self.feature_keys)  # [F, S, S]

            # ---- cache save (always store full feature set) ----
            if use_cache:
                self.cache.save_features(pair[0], pair[1], {k: v.tolist() for k, v in feature_map.items()})

            tensors.append(stacked)

        # [N, F, S, S]
        self.features = torch.stack(tensors, dim=0)
        self.labels   = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

        # Length scalars: [log(n+1), log(m+1)] per pair — used by length-conditioned model.
        # split_txt with resolve_coref=False is pure regex so this loop is fast.
        lengths_list = [
            [math.log(len(split_txt(p[0])) + 1), math.log(len(split_txt(p[1])) + 1)]
            for p in paragraph_pairs
        ]
        self.lengths = torch.tensor(lengths_list, dtype=torch.float32)  # [N, 2]

        print(f"Feature tensor shape : {self.features.shape}")
        print(f"Labels shape         : {self.labels.shape}")

        # Free extractor models — called as a separate method so its local
        # imports live in their own scope and cannot shadow any name used above.
        self._free_extractor_models()

    def _free_extractor_models(self) -> None:
        """Free all feature-extractor model weights from RAM.

        Called once after self.features is fully constructed. Clears both the
        extractor instances on self and the class-level model caches that hold
        the actual weight tensors (~5–8 GB total). Frees memory before the CNN
        training loop starts.

        Implemented as a separate method — NOT inline in __init__ — so that
        the ``import X as alias`` statements here live in their own local scope
        and cannot cause Python to treat module-level names (e.g. NLIWeights)
        as unbound locals in __init__.
        """
        import gc

        del self.lexical, self.semantic, self.nli, self.entity
        del self.lcs, self.numeric, self.relations, self.relex

        try:
            import backend.Features.Semantic.__generate_semantic_features as _sem
            _sem.SemanticFeatures._model_cache.clear()
        except Exception:
            pass
        try:
            import backend.Features.NLI.getNLIweights as _nli
            _nli.NLIWeights._model = None
            _nli.NLIWeights._tokenizer = None
            _nli.NLIWeights._disk_cache_loaded = False
        except Exception:
            pass
        try:
            import backend.Features.EntityGroups.getOverlap as _ent
            _ent.EntityMatch._model_cache.clear()
            _ent.EntityMatch._disk_cache_loaded = False
        except Exception:
            pass
        try:
            import backend.Features.Relations.getRelexWeights as _relex
            _relex.RelexGrounding._model_cache.clear()
            _relex.RelexGrounding._disk_cache_loaded = False
        except Exception:
            pass

        gc.collect()
        print("[MemClean] Feature extractor models freed — starting CNN training loop.")

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.lengths[idx], self.labels[idx]


def train_model(model, train_loader, val_loader, num_epochs=75, learning_rate=0.0003,
                patience=15, min_delta=0.001, best_ckpt='best_model.pth', mode='general',
                label_smooth=0.05, warmup_epochs=5, feature_keys: list[str] | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_features = train_loader.dataset.num_features

    training_params = {
        'num_epochs':                num_epochs,
        'learning_rate':             learning_rate,
        'batch_size':                train_loader.batch_size,
        'optimizer':                 'AdamW',
        'weight_decay':              1e-4,
        'lr_schedule':               f'LinearWarmup({warmup_epochs}ep)+CosineAnnealingLR',
        'loss_function':             'MSELoss',
        'label_smooth':              label_smooth,
        'warmup_epochs':             warmup_epochs,
        'early_stopping_patience':   patience,
        'early_stopping_min_delta':  min_delta,
        'num_feature_channels':      num_features,
        'use_length_cond':           getattr(model, 'use_length_cond', False),
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Warmup: ramp from lr*0.1 → lr over warmup_epochs, then cosine decay to eta_min.
    # Longer schedule (75ep default) + wider patience (15) prevents early stopping on val noise.
    _warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    _cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs - warmup_epochs, 1), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[_warmup, _cosine], milestones=[warmup_epochs]
    )

    best_val_loss      = float('inf')
    patience_counter   = 0
    best_model_state   = None
    best_optimizer_state = None
    best_epoch         = 0

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for features, lengths, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features = features.to(device)
            lengths  = lengths.to(device)
            labels   = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features, lengths)
            # Label smoothing: map 0→ε, 1→(1−ε) to soften confident targets on noisy data
            smooth_labels = labels * (1 - 2 * label_smooth) + label_smooth if label_smooth > 0 else labels
            loss    = criterion(outputs, smooth_labels)
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
            for features, lengths, labels in val_loader:
                features = features.to(device)
                lengths  = lengths.to(device)
                labels   = labels.to(device)
                outputs   = model(features, lengths)
                val_loss += criterion(outputs, labels).item()
                predicted  = (outputs > 0.5).float()
                total     += labels.size(0)
                correct   += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy     = 100 * correct / total

        report.update_epoch_metrics(epoch + 1, avg_train_loss, avg_val_loss, accuracy)
        tracker.log_epoch(epoch + 1, avg_train_loss, avg_val_loss, accuracy)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f'Epoch [{epoch+1}/{num_epochs}]  '
              f'Train: {avg_train_loss:.4f}  Val: {avg_val_loss:.4f}  Acc: {accuracy:.2f}%  LR: {current_lr:.6f}')

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
                'hidden_dim':           model.conv1.out_channels,
                'spatial_size':         SPATIAL_SIZE,
                'use_length_cond':      getattr(model, 'use_length_cond', False),
                'manifest':             build_manifest(feature_keys),
            }, best_ckpt)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs "
                  f"(best val loss {best_val_loss:.4f} @ epoch {best_epoch + 1})")
            break

    # --- Save final artefacts ---
    ts        = report.timestamp
    ckpt_dir  = Path(best_ckpt).parent   # models/{mode}/ or '.' for general
    final_model_path = str(ckpt_dir / f'{ts}_final.pth')
    best_model_path  = str(ckpt_dir / f'{ts}_best.pth')

    manifest = build_manifest(feature_keys)

    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss':           avg_val_loss,
        'best_loss':            best_val_loss,
        'final_accuracy':       accuracy,
        'num_features':         num_features,
        'hidden_dim':           model.conv1.out_channels,
        'spatial_size':         SPATIAL_SIZE,
        'use_length_cond':      getattr(model, 'use_length_cond', False),
        'manifest':             manifest,
    }, final_model_path)

    torch.save({
        'epoch':                best_epoch,
        'model_state_dict':     best_model_state,
        'optimizer_state_dict': best_optimizer_state,
        'loss':                 best_val_loss,
        'num_features':         num_features,
        'hidden_dim':           model.conv1.out_channels,
        'spatial_size':         SPATIAL_SIZE,
        'use_length_cond':      getattr(model, 'use_length_cond', False),
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
        from pathlib import Path as _Path
        data_dir  = f'data/{args.mode}'
        mode_dir  = _Path('models') / args.mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = str(mode_dir / 'best.pth')
        print(f"Mode: {args.mode}  |  Data: {data_dir}/  |  Checkpoint: {best_ckpt}")
    else:
        data_dir  = 'data'
        best_ckpt = 'best_model.pth'
        print("Mode: general  |  Data: data/  |  Checkpoint: best_model.pth")

    # Resolve mode-specific feature basket (v5.0+).
    active_feature_keys = get_feature_keys(args.mode)
    n_features = len(active_feature_keys)
    print(f"Feature basket : {n_features} features  {active_feature_keys}")

    train_pairs, train_labels = load_json_data(f'{data_dir}/train.json')
    val_pairs,   val_labels   = load_json_data(f'{data_dir}/validate.json')
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    train_dataset = TextSimilarityDataset(train_pairs, train_labels, use_cache=True,
                                          feature_keys=active_feature_keys)
    val_dataset   = TextSimilarityDataset(val_pairs,   val_labels,   use_cache=True,
                                          feature_keys=active_feature_keys)

    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader    = DataLoader(val_dataset,   batch_size=16)

    model = TextSimilarityCNN(num_features=train_dataset.num_features, spatial_size=SPATIAL_SIZE,
                               use_length_cond=False)
    trained_model, metrics = train_model(
        model, train_loader, val_loader, best_ckpt=best_ckpt,
        mode=args.mode or "general",
        label_smooth=0.0,
        feature_keys=active_feature_keys,
    )
    print("Training complete.")
