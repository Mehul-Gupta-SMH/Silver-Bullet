import torch
import argparse
import json
from model import TextSimilarityCNN
from train import TextSimilarityDataset
from pathlib import Path
import numpy as np

class SimilarityPredictor:
    def __init__(self, model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)

        # Get input dimension from the first layer of the model
        # We'll create a temporary dataset to get the feature dimension
        temp_dataset = TextSimilarityDataset([["temp", "temp"]], [0], use_cache=True)
        input_dim = temp_dataset.features.shape[1]

        # Initialize and load model
        self.model = TextSimilarityCNN(input_dim=input_dim)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Model was trained for {self.checkpoint['epoch']} epochs")
        if 'accuracy' in self.checkpoint:
            print(f"Model validation accuracy: {self.checkpoint['accuracy']:.2f}%")

    def predict_pair(self, text1, text2):
        """Make prediction for a single pair of texts"""
        # Create dataset with single pair
        dataset = TextSimilarityDataset([[text1, text2]], [0], use_cache=True)
        features = dataset.features

        # Make prediction
        with torch.no_grad():
            features = features.to(self.device)
            output = self.model(features)
            prob = output.item()
            prediction = 1 if prob >= 0.5 else 0

        return {
            'prediction': prediction,
            'probability': prob,
            'text1': text1,
            'text2': text2
        }

    def predict_batch(self, pairs):
        """Make predictions for a batch of text pairs"""
        # Create dummy labels (they won't be used for prediction)
        dummy_labels = [0] * len(pairs)

        # Create dataset with all pairs
        dataset = TextSimilarityDataset(pairs, dummy_labels, use_cache=True)
        features = dataset.features

        # Make predictions
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            probs = outputs.cpu().numpy().flatten()
            predictions = (probs >= 0.5).astype(int)

        # Prepare results
        results = []
        for i, (pair, pred, prob) in enumerate(zip(pairs, predictions, probs)):
            results.append({
                'prediction': int(pred),
                'probability': float(prob),
                'text1': pair[0],
                'text2': pair[1]
            })

        return results

def save_predictions(predictions, output_file):
    """Save predictions to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on text pairs using trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file containing text pairs')
    parser.add_argument('--output', type=str, default='predictions.json', help='Path to save predictions')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to run inference on')
    args = parser.parse_args()

    # Load input data
    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # Convert input data to pairs
    if isinstance(input_data, list):
        # Assume list of pairs
        pairs = input_data
    elif isinstance(input_data, dict) and 'data' in input_data:
        # Assume format like training data
        pairs = [[item['text1'], item['text2']] for item in input_data['data']]
    else:
        raise ValueError("Input JSON must be either a list of pairs or a dict with 'data' key")

    # Initialize predictor
    predictor = SimilarityPredictor(args.model, device=args.device)

    # Make predictions
    print(f"Making predictions for {len(pairs)} pairs...")
    predictions = predictor.predict_batch(pairs)

    # Save predictions
    save_predictions(predictions, args.output)

    # Print summary
    positive_preds = sum(1 for p in predictions if p['prediction'] == 1)
    print(f"\nPrediction Summary:")
    print(f"Total pairs processed: {len(predictions)}")
    print(f"Similar pairs found: {positive_preds}")
    print(f"Non-similar pairs found: {len(predictions) - positive_preds}")

if __name__ == '__main__':
    main()
