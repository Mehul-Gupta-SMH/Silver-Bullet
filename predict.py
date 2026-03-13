import torch
import argparse
import json
from model import TextSimilarityCNN, TextSimilarityCNNLegacy
from train import TextSimilarityDataset, feature_map_to_tensor
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
        # Current Conv2D checkpoint — num_features must be present
        if 'num_features' not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'num_features'. "
                "Re-train the model to generate a compatible checkpoint."
            )
        num_features = checkpoint['num_features']
        model = TextSimilarityCNN(num_features=num_features)
        arch = f'Conv2D (num_features={num_features})'

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

        Unlike predict_pair(), this method runs the feature extractors manually so
        that intermediate per-sentence data (alignment matrix, feature maps) can be
        inspected.  Model caching on the feature-extractor classes means the heavy
        models are only loaded once per process.

        Returns a dict compatible with api.schemas.BreakdownResponse.
        """
        from Splitter.sentence_splitter import split_txt
        from Features.Semantic.getSemanticWeights import SemanticWeights
        from Features.Lexical.getLexicalWeights import LexicalWeights
        from Features.NLI.getNLIweights import NLIWeights
        from Features.EntityGroups.getOverlap import EntityMatch
        from Features.LCS.getLCSweights import LCSWeights

        sent1 = split_txt(text1)
        sent2 = split_txt(text2)
        n, m = len(sent1), len(sent2)

        feature_map: dict = {}
        feature_map.update(LexicalWeights().getFeatureMap(sent1, sent2))
        feature_map.update(SemanticWeights().getFeatureMap(sent1, sent2))
        feature_map.update(NLIWeights().getFeatureMap(sent1, sent2))
        feature_map.update(EntityMatch().getFeatureMap(sent1, sent2))
        feature_map.update(LCSWeights().getFeatureMap(sent1, sent2))

        # --- model score ---
        stacked = feature_map_to_tensor(feature_map).unsqueeze(0)  # [1, F, 64, 64]
        if isinstance(self.model, TextSimilarityCNNLegacy):
            trained_num_maps = self.model.fc_reduce1.in_features // (64 * 64)
            stacked = stacked[:, :trained_num_maps, :, :]
            stacked = stacked.view(1, -1)

        with torch.no_grad():
            prob = self.model(stacked.to(self.device)).item()

        # --- alignment matrix: mxbai cosine sim cropped to actual sentence count ---
        sem_key = "mixedbread-ai/mxbai-embed-large-v1"
        alignment: list[list[float]] = []
        if n > 0 and m > 0 and sem_key in feature_map:
            alignment = feature_map[sem_key].numpy()[:n, :m].tolist()

        # --- divergence (sentences with no good counterpart) ---
        THRESH = 0.5
        divergent_in_1: list[int] = []
        divergent_in_2: list[int] = []
        if alignment:
            max_row = [max(row) for row in alignment]
            max_col = [max(alignment[i][j] for i in range(n)) for j in range(m)]
            divergent_in_1 = [i for i, s in enumerate(max_row) if s < THRESH]
            divergent_in_2 = [j for j, s in enumerate(max_col) if s < THRESH]

        # --- per-feature group scores: mean best-match score across sentences in text1 ---
        def _mean_max_row(key: str) -> float:
            if key not in feature_map or n == 0 or m == 0:
                return 0.0
            mat = feature_map[key].numpy()[:n, :m]
            return float(np.mean([mat[i].max() for i in range(n)]))

        feature_scores = {
            "Semantic (mxbai)": _mean_max_row("mixedbread-ai/mxbai-embed-large-v1"),
            "Semantic (Qwen3)": _mean_max_row("Qwen/Qwen3-Embedding-0.6B"),
            "Lexical ROUGE":    _mean_max_row("rouge"),
            "Lexical Jaccard":  _mean_max_row("jaccard"),
            "NLI Entailment":   _mean_max_row("entailment"),
            "LCS Token":        _mean_max_row("lcs_token"),
        }

        return {
            "prediction":     int(prob >= 0.5),
            "probability":    prob,
            "sentences1":     sent1,
            "sentences2":     sent2,
            "alignment":      alignment,
            "divergent_in_1": divergent_in_1,
            "divergent_in_2": divergent_in_2,
            "feature_scores": feature_scores,
        }

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
