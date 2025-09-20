 # Silver Bullet
------
<img src="assets/icon.png" alt="Project Banner" width="100%" style="max-width:600px; height:auto;">

Silver Bullet is an evaluation framework that takes inspiration from Reward models used in Reinforcement Learning, especially in RLHF.
The framework helps the user, based on the two texts to be compared, answer pertinent questions:
 1. Question and Answer pair: Can be used to confirm if the answer covers points about the question
 2. Answer and Answer pair: Compare human preference for one of the answers
 3. Context and Answer pair: Compare answer faithfulness

The final result is a score that depends upon pairs and use cases.
e.g: For answer and answer pair, the score can be interpreted that the answers are comparable or just slightly different, but one of the model is cheaper to use. This can effectively be used to choose models and compare performance.

-----------------

## Features
- **Lexical Analysis**: Measures surface-level similarity using word overlap and related metrics.
- **Semantic Analysis**: Uses transformer-based models to assess deeper meaning and context similarity.
- **NLI Analysis**: Evaluates logical relationships (entailment, contradiction, neutrality) between sentences.
- **Entity Counts**: Evaluates Entity counts between sentences to get soft idea for entity overlaps.
- **Sentence Splitting**: Breaks paragraphs into sentences for granular comparison.
- **Postprocessing**: Pads feature matrices for consistent output shapes.
- **Feature Caching**: Stores computed features for faster subsequent runs.
- **Comprehensive Training Reports**: Generates detailed reports and metrics during training.

## Model Architecture
The framework includes a CNN-based model for text similarity classification:
- Multi-stage dimensionality reduction (57344 → 4096 → 1024 → 256)
- Convolutional layers for feature extraction
- Batch normalization and dropout for regularization
- Binary classification output (0-1 similarity score)

-----------------

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create required directories:
   ```bash
   mkdir -p data cache training_reports
   ```

-----------------

## Usage

### Feature Extraction
```bash
python example.py
```
This will compare example paragraphs and print feature maps for lexical, semantic, and NLI similarity.

### Training
1. Prepare your training data in JSON format (see `data/train.json` for example)
2. Run the training script:
```bash
python train.py
```

Features during training:
- Automatic feature caching for faster subsequent runs
- Early stopping with configurable patience
- Comprehensive training reports and metrics
- Both best and final model weights are saved
- Real-time training progress monitoring

### Testing
To evaluate the model on new data:
```bash
python test.py
```
This will generate detailed test results including:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Individual predictions for each test pair

### Inference
To run inference on new text pairs:
```bash
python predict.py --model path/to/best_model.pth --input input.json --output predictions.json
```

The input JSON file can be in one of two formats:
1. List of text pairs:
```json
[
    ["text1", "text2"],
    ["text3", "text4"]
]
```

2. Dictionary format (same as training data):
```json
{
    "data": [
        {
            "text1": "First text",
            "text2": "Second text"
        }
    ]
}
```

Optional arguments:
- `--device`: Specify 'cuda' or 'cpu' for inference device
- `--output`: Custom path for predictions output (default: predictions.json)

The script will:
- Load the trained model
- Process text pairs using the same feature extraction pipeline
- Utilize feature caching for efficiency
- Generate predictions with confidence scores
- Save results in JSON format

Output format:
```json
[
    {
        "prediction": 1,
        "probability": 0.92,
        "text1": "input text 1",
        "text2": "input text 2"
    }
]
```

## Project Structure
- `model.py` — CNN model architecture for text similarity
- `train.py` — Training pipeline with feature caching and reporting
- `test.py` — Model evaluation and testing
- `example.py` — Feature extraction demonstration
- `training_report.py` — Training metrics and report generation
- `feature_cache.py` — Feature caching implementation
- `requirements.txt` — Python dependencies
- `Features/`
  - `Lexical/` — Lexical feature extraction
  - `Semantic/` — Semantic feature extraction
  - `NLI/` — NLI feature extraction
  - `EntityGroups/` — Entity overlap analysis
- `Splitter/` — Sentence splitting utility
- `Postprocess/` — Matrix padding utility
- `data/` — Training, validation, and test data
- `cache/` — Cached features storage
- `training_reports/` — Training reports and metrics

## Training Reports
The system generates comprehensive training reports including:
- Model architecture details
- Training parameters
- Dataset information
- Hardware configuration
- Training metrics (losses, accuracies)
- Best model performance
- Early stopping information

Reports are saved in both JSON and Markdown formats in the `training_reports` directory.

## Model Weights
The system saves multiple model checkpoints:
- `best_model.pth` — Best performing model during training
- `model_weights_[timestamp]_best.pth` — Best weights with full training context
- `model_weights_[timestamp]_final.pth` — Final model state after training

## Extending
You can extend the framework by:
1. Adding new feature extractors in the `Features/` directory
2. Modifying the model architecture in `model.py`
3. Customizing training parameters in `train.py`
4. Adding new metrics to the training reports

## License
Specify your license here.

---
*For questions or contributions, please open an issue or pull request.*
