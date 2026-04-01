import torch
import argparse
import json
from backend.model import TextSimilarityCNN, TextSimilarityCNNLegacy
from backend.train import TextSimilarityDataset, feature_map_to_tensor
from backend.feature_registry import validate_manifest
from pathlib import Path
import numpy as np


def _load_model_from_checkpoint(checkpoint, device):
    """Auto-detect architecture from checkpoint keys and return an initialised model."""
    state = checkpoint['model_state_dict']

    if 'fc_reduce1.weight' in state:
        # Legacy Conv1D checkpoint — derive input_dim from the first FC layer weight
        input_dim = state['fc_reduce1.weight'].shape[1]
        model = TextSimilarityCNNLegacy(input_dim=input_dim)
        arch = f'legacy Conv1D (input_dim={input_dim})'
    else:
        # Current Conv2D checkpoint — validate manifest then load
        if 'num_features' not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'num_features'. "
                "Re-train the model to generate a compatible checkpoint."
            )
        validate_manifest(checkpoint.get('manifest'))
        num_features = checkpoint['num_features']
        spatial_size = checkpoint.get('spatial_size', 64)  # 64 = legacy default
        hidden_dim   = checkpoint.get('hidden_dim', 128)   # 128 = legacy default
        model = TextSimilarityCNN(num_features=num_features, spatial_size=spatial_size, hidden_dim=hidden_dim)
        arch = f'Conv2D (num_features={num_features}, spatial_size={spatial_size}, hidden_dim={hidden_dim})'

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, arch


class SimilarityPredictor:
    def __init__(self, model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model, arch = _load_model_from_checkpoint(checkpoint, self.device)

        print(f"Model loaded from {model_path}")
        print(f"  architecture : {arch}")
        print(f"  epoch        : {checkpoint.get('epoch', 'unknown')}")
        if 'accuracy' in checkpoint:
            print(f"  val acc      : {checkpoint['accuracy']:.2f}%")

    def _extract_features(self, pairs):
        """Run the full feature pipeline and return tensors shaped for the loaded model.

        Conv2D model: [N, F, 64, 64]
        Legacy Conv1D model: [N, F*64*64]  (flattened)
        """
        dummy_labels = [0] * len(pairs)
        dataset = TextSimilarityDataset(pairs, dummy_labels, use_cache=True)
        features = dataset.features  # always [N, F, 64, 64]
        if isinstance(self.model, TextSimilarityCNNLegacy):
            # Truncate to the number of feature maps the legacy model was trained on.
            # The checkpoint input_dim encodes exactly how many maps were used:
            # input_dim = num_maps * 64 * 64.  New maps (e.g. LCS) are appended last
            # by the feature pipeline, so slicing the channel dimension is safe.
            trained_num_maps = self.model.fc_reduce1.in_features // (64 * 64)
            features = features[:, :trained_num_maps, :, :]
            features = features.view(features.size(0), -1)
        return features

    def predict_pair(self, text1, text2):
        """Make prediction for a single pair of texts."""
        features = self._extract_features([[text1, text2]])

        with torch.no_grad():
            output = self.model(features.to(self.device))
            prob = output.item()

        return {
            'prediction': int(prob >= 0.5),
            'probability': prob,
            'text1': text1,
            'text2': text2,
        }

    def predict_pair_breakdown(self, text1: str, text2: str) -> dict:
        """Run the full pipeline and return per-sentence alignment + feature group scores.

        Checks the feature cache first (populated by predict_pair / training).  On a
        cache miss the five extractors run in parallel via ThreadPoolExecutor and the
        result is saved so subsequent calls for the same pair are instant.

        Returns a dict compatible with api.schemas.BreakdownResponse.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from backend.Splitter.sentence_splitter import split_txt
        from backend.Features.Semantic.getSemanticWeights import SemanticWeights
        from backend.Features.Lexical.getLexicalWeights import LexicalWeights
        from backend.Features.NLI.getNLIweights import NLIWeights
        from backend.Features.EntityGroups.getOverlap import EntityMatch
        from backend.Features.LCS.getLCSweights import LCSWeights
        from backend.feature_cache import FeatureCache

        sent1 = split_txt(text1)
        sent2 = split_txt(text2)
        n, m = len(sent1), len(sent2)

        # --- try cache first ---
        cache = FeatureCache()
        cached = cache.get_features(text1, text2)
        if cached is not None and isinstance(cached, dict):
            feature_map = {k: torch.tensor(v, dtype=torch.float32) for k, v in cached.items()}
        else:
            # --- parallel feature extraction on cache miss ---
            extractors = [
                ("lexical",  LexicalWeights),
                ("semantic", SemanticWeights),
                ("nli",      NLIWeights),
                ("entity",   EntityMatch),
                ("lcs",      LCSWeights),
            ]

            def _run(name_cls):
                name, cls = name_cls
                return cls().getFeatureMap(sent1, sent2)

            feature_map: dict = {}
            with ThreadPoolExecutor(max_workers=len(extractors)) as pool:
                futures = {pool.submit(_run, item): item[0] for item in extractors}
                for fut in as_completed(futures):
                    feature_map.update(fut.result())

            # persist so next call (or predict_pair on the same pair) hits cache
            cache.save_features(text1, text2, {k: v.tolist() for k, v in feature_map.items()})

        # --- model score ---
        stacked = feature_map_to_tensor(feature_map).unsqueeze(0)  # [1, F, S, S]
        if isinstance(self.model, TextSimilarityCNNLegacy):
            trained_num_maps = self.model.fc_reduce1.in_features // (64 * 64)
            stacked = stacked[:, :trained_num_maps, :, :]
            stacked = stacked.view(1, -1)

        with torch.no_grad():
            prob = self.model(stacked.to(self.device)).item()

        # --- alignment matrix: mxbai cosine sim cropped to actual sentence count ---
        # Feature maps are resized to SPATIAL_SIZE×SPATIAL_SIZE, so clamp n/m to that.
        from backend.Postprocess.__addpad import TARGET_SIZE as _SPATIAL
        n_crop, m_crop = min(n, _SPATIAL), min(m, _SPATIAL)
        sem_key = "mixedbread-ai/mxbai-embed-large-v1"
        alignment: list[list[float]] = []
        if n_crop > 0 and m_crop > 0 and sem_key in feature_map:
            alignment = feature_map[sem_key].numpy()[:n_crop, :m_crop].tolist()

        # --- divergence (sentences with no good counterpart) ---
        THRESH = 0.5
        divergent_in_1: list[int] = []
        divergent_in_2: list[int] = []
        if alignment:
            max_row = [max(row) for row in alignment]
            max_col = [max(alignment[i][j] for i in range(n_crop)) for j in range(m_crop)]
            divergent_in_1 = [i for i, s in enumerate(max_row) if s < THRESH]
            divergent_in_2 = [j for j, s in enumerate(max_col) if s < THRESH]

        # --- per-feature group scores: mean best-match score across sentences in text1 ---
        def _mean_max_row(key: str) -> float:
            if key not in feature_map or n_crop == 0 or m_crop == 0:
                return 0.0
            mat = feature_map[key].numpy()[:n_crop, :m_crop]
            return float(np.mean([mat[i].max() for i in range(n_crop)]))

        # EntityMismatch scores are <= 0 (sum of -abs(count_diff) per entity type).
        # 0 = perfect entity match, more negative = more mismatch.
        # exp(mean_score) maps (-inf, 0] -> (0, 1] giving an intuitive [0,1] signal.
        def _entity_match_score(key: str) -> float:
            if key not in feature_map or n_crop == 0 or m_crop == 0:
                return 1.0  # no entities detected → no mismatch
            mat = feature_map[key].numpy()[:n_crop, :m_crop]
            return float(np.exp(float(mat.mean())))

        feature_scores = {
            "Semantic (mxbai)": _mean_max_row("mixedbread-ai/mxbai-embed-large-v1"),
            "Semantic (Qwen3)": _mean_max_row("Qwen/Qwen3-Embedding-0.6B"),
            "Lexical ROUGE":    _mean_max_row("rouge"),
            "Lexical Jaccard":  _mean_max_row("jaccard"),
            "NLI Entailment":   _mean_max_row("entailment"),
            "Entity Match":     _entity_match_score("EntityMismatch"),
            "LCS Token":        _mean_max_row("lcs_token"),
        }

        misalignment_reasons = _generate_misalignment_reasons(
            prob, feature_scores, divergent_in_1, divergent_in_2, sent1, sent2,
        )

        return {
            "prediction":           int(prob >= 0.5),
            "probability":          prob,
            "sentences1":           sent1,
            "sentences2":           sent2,
            "alignment":            alignment,
            "divergent_in_1":       divergent_in_1,
            "divergent_in_2":       divergent_in_2,
            "feature_scores":       feature_scores,
            "misalignment_reasons": misalignment_reasons,
        }

    def predict_batch_breakdown(self, pairs: list[list[str]]) -> list[dict]:
        """Run predict_pair_breakdown for each pair in the batch."""
        return [self.predict_pair_breakdown(pair[0], pair[1]) for pair in pairs]

    def predict_batch(self, pairs):
        """Make predictions for a list of [text1, text2] pairs."""
        features = self._extract_features(pairs)

        with torch.no_grad():
            outputs = self.model(features.to(self.device))
            probs = outputs.cpu().numpy().flatten()
            predictions = (probs >= 0.5).astype(int)

        return [
            {
                'prediction': int(pred),
                'probability': float(prob),
                'text1': pair[0],
                'text2': pair[1],
            }
            for pair, pred, prob in zip(pairs, predictions, probs)
        ]


def _generate_misalignment_reasons(
    probability: float,
    feature_scores: dict,
    divergent_in_1: list,
    divergent_in_2: list,
    sentences1: list,
    sentences2: list,
) -> list:
    """Return a ranked list of reason dicts explaining misalignment signals.

    Each reason: {label, description, severity ('high'|'medium'|'low'), signal}
    Rules fire on individual feature thresholds; results sorted high→medium→low.
    Reasons are generated regardless of overall score so callers get diagnostic
    context even for borderline-positive pairs.
    """
    reasons: list[dict] = []

    sem_mxbai  = feature_scores.get("Semantic (mxbai)", 1.0)
    sem_qwen   = feature_scores.get("Semantic (Qwen3)", 1.0)
    rouge      = feature_scores.get("Lexical ROUGE", 1.0)
    jaccard    = feature_scores.get("Lexical Jaccard", 1.0)
    entailment = feature_scores.get("NLI Entailment", 1.0)
    entity     = feature_scores.get("Entity Match", 1.0)
    lcs        = feature_scores.get("LCS Token", 1.0)

    fired: set[str] = set()

    def _add(label, description, severity, signal):
        reasons.append({"label": label, "description": description,
                         "severity": severity, "signal": signal})
        fired.add(label)

    # ── High ──────────────────────────────────────────────────────────────────
    if entailment < 0.25:
        _add(
            "Entailment Conflict",
            "The NLI model finds no strong entailment between texts. Key claims in text1 "
            "may be contradicted, negated, or entirely absent in text2.",
            "high", "NLI Entailment",
        )

    if sem_mxbai < 0.35 and sem_qwen < 0.35:
        _add(
            "Semantic Divergence",
            "Both embedding models report low sentence-level similarity. The texts likely "
            "convey fundamentally different information.",
            "high", "Semantic (mxbai + Qwen3)",
        )

    if entity < 0.45:
        _add(
            "Entity Substitution",
            "Named entities (people, numbers, dates, locations) differ significantly between "
            "texts — a hallmark of factual hallucination in generation contexts.",
            "high", "Entity Match",
        )

    # ── Medium ────────────────────────────────────────────────────────────────
    if 0.25 <= entailment < 0.40 and "Entailment Conflict" not in fired:
        _add(
            "Weak Entailment",
            "Entailment signal is below expected threshold. Text2 may not fully support all "
            "claims in text1, or the logical relationship between texts is ambiguous.",
            "medium", "NLI Entailment",
        )

    if 0.45 <= entity < 0.68 and "Entity Substitution" not in fired:
        _add(
            "Partial Entity Overlap",
            "Some named entities are substituted or missing. Verify specific numbers, "
            "proper nouns, and factual references between the two texts.",
            "medium", "Entity Match",
        )

    if rouge < 0.25 and jaccard < 0.20:
        _add(
            "Low Lexical Overlap",
            "Very few shared tokens between texts. This may indicate a complete topic switch "
            "or aggressive paraphrasing that alters the original meaning.",
            "medium", "Lexical ROUGE + Jaccard",
        )

    if divergent_in_1:
        n_div, total = len(divergent_in_1), len(sentences1)
        _add(
            f"Uncovered Claims ({n_div}/{total} sentences)",
            f"{n_div} of {total} sentence(s) in text1 have no semantically similar counterpart "
            "in text2. These claims may be omitted or left unsupported.",
            "medium", "Alignment",
        )

    if divergent_in_2:
        n_div, total = len(divergent_in_2), len(sentences2)
        _add(
            f"Unanchored Content ({n_div}/{total} sentences)",
            f"{n_div} of {total} sentence(s) in text2 have no strong counterpart in text1. "
            "Text2 may contain fabricated or hallucinated information.",
            "medium", "Alignment",
        )

    # ── Low ───────────────────────────────────────────────────────────────────
    if (sem_mxbai >= 0.50 and entailment < 0.35
            and "Entailment Conflict" not in fired
            and "Weak Entailment" not in fired):
        _add(
            "Structural Mismatch",
            "Texts are semantically close but the logical relationship is weak — "
            "same topic but possibly different conclusions or framing.",
            "low", "NLI Entailment",
        )

    if rouge < 0.30 and sem_mxbai >= 0.55:
        _add(
            "Abstractive Paraphrase Risk",
            "High semantic similarity but low lexical overlap suggests heavy paraphrasing. "
            "Abstractive rewrites can introduce subtle meaning shifts.",
            "low", "Lexical ROUGE",
        )

    if lcs < 0.20 and probability < 0.5:
        _add(
            "Low Sequential Overlap",
            "The longest common subsequence of tokens is very short. "
            "Texts share little sequential content structure.",
            "low", "LCS Token",
        )

    # Sort: high → medium → low
    _order = {"high": 0, "medium": 1, "low": 2}
    reasons.sort(key=lambda r: _order[r["severity"]])
    return reasons


def save_predictions(predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on text pairs using trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, default='predictions.json', help='Output JSON path')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Inference device')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    if isinstance(input_data, list):
        pairs = input_data
    elif isinstance(input_data, dict) and 'data' in input_data:
        pairs = [[item['text1'], item['text2']] for item in input_data['data']]
    else:
        raise ValueError("Input JSON must be a list of pairs or a dict with a 'data' key.")

    predictor = SimilarityPredictor(args.model, device=args.device)

    print(f"Making predictions for {len(pairs)} pairs...")
    predictions = predictor.predict_batch(pairs)

    save_predictions(predictions, args.output)

    positive = sum(1 for p in predictions if p['prediction'] == 1)
    print(f"\nPrediction Summary:")
    print(f"  Total:     {len(predictions)}")
    print(f"  Similar:   {positive}")
    print(f"  Different: {len(predictions) - positive}")


if __name__ == '__main__':
    main()
