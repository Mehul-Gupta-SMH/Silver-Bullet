Silver Bullet
------
-----------------
Silver Bullet is a Python toolkit for comparing two paragraphs or documents using advanced Natural Language Processing (NLP) techniques. It extracts and compares features at the lexical, semantic, and natural language inference (NLI) levels, providing a detailed similarity analysis between text segments.

-----------------

## Features
- **Lexical Analysis**: Measures surface-level similarity using word overlap and related metrics.
- **Semantic Analysis**: Uses transformer-based models to assess deeper meaning and context similarity.
- **NLI Analysis**: Evaluates logical relationships (entailment, contradiction, neutrality) between sentences.
- **Sentence Splitting**: Breaks paragraphs into sentences for granular comparison.
- **Postprocessing**: Pads feature matrices for consistent output shapes.

-----------------

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

-----------------

## Usage
Run the main script to see a demonstration:
```bash
python example.py
```
This will compare two example paragraphs and print feature maps for lexical, semantic, and NLI similarity.

## Project Structure
- `main.py` — Entry point; demonstrates feature extraction and comparison.
- `requirements.txt` — Python dependencies.
- `Features/`
  - `Lexical/`
    - `getLexicalWeights.py` — Lexical feature extraction.
  - `Semantic/`
    - `getSemanticWeights.py` — Semantic feature extraction.
  - `NLI/`
    - `getNLIweights.py` — NLI feature extraction.
- `Splitter/`
  - `sentence_splitter.py` — Sentence splitting utility.
- `Postprocess/`
  - `_addpad.py` — Matrix padding utility.
- `Preprocess/` — (Reserved for preprocessing utilities, e.g., coreference resolution.)

## Extending
You can add new feature extraction methods by creating new modules in the `Features/` directory and integrating them in `main.py`.

## License
Specify your license here.

---
*For questions or contributions, please open an issue or pull request.*

