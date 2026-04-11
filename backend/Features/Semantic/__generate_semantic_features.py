import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm

# One cache file per model, stored in cache/embeddings_{sanitised_name}/
# Format: numpy .npz with two arrays:
#   "sentences" — 1-D array of sentence strings (object dtype)
#   "embeddings" — 2-D float32 array, shape (N, dim)
_EMBED_CACHE_DIR = Path("cache/embeddings")


def _cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "__").replace("\\", "__")
    return _EMBED_CACHE_DIR / f"{safe}.npz"


class SemanticFeatures:
    # Shared across all instances: loaded SentenceTransformer objects
    _model_cache = {}
    # Shared across all instances: model_name → {sentence: np.ndarray}
    _embedding_cache: dict = {}
    # Guard: set of model names whose disk cache has been loaded this process
    _disk_loaded: set = set()

    def __init__(self, text_list: List[str] = None):
        self.feature_model_local_list = [
            {
                'model':   "mixedbread-ai/mxbai-embed-large-v1",
                'task':    "text-matching",
                'prompt':  "text-matching",
                'pooling': False
            },
            {
                'model':   "Qwen/Qwen3-Embedding-0.6B",
                'task':    "text-matching",
                'prompt':  "text-matching",
                'pooling': False
            }
        ]

        self.text_list = text_list or []
        self.features_dict = {}

    def set_sentences(self, text_list: List[str]):
        """Allow reusing the same object with different text inputs."""
        self.text_list = text_list
        self.features_dict = {}

    # ------------------------------------------------------------------
    # Persistent embedding cache — load / save
    # ------------------------------------------------------------------

    @classmethod
    def load_embedding_cache(cls, model_name: str) -> None:
        """Load persisted embeddings for *model_name* from disk (once per process)."""
        if model_name in cls._disk_loaded:
            return
        cls._disk_loaded.add(model_name)

        path = _cache_path(model_name)
        if not path.exists():
            return
        try:
            data = np.load(path, allow_pickle=True)
            sentences = data["sentences"].tolist()   # list of str
            embeddings = data["embeddings"]           # (N, dim) float32
            sent_cache = cls._embedding_cache.setdefault(model_name, {})
            for s, e in zip(sentences, embeddings):
                sent_cache[s] = e
            print(f"[EmbedCache] loaded {len(sentences)} embeddings for {model_name}")
        except Exception as ex:
            print(f"[EmbedCache] warning: could not load cache for {model_name} ({ex}) — starting fresh")

    @classmethod
    def save_embedding_cache(cls, model_name: str) -> None:
        """Persist current in-memory embeddings for *model_name* to disk."""
        sent_cache = cls._embedding_cache.get(model_name, {})
        if not sent_cache:
            return
        _EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sentences = list(sent_cache.keys())
        embeddings = np.stack([sent_cache[s] for s in sentences]).astype(np.float32)
        np.savez_compressed(
            _cache_path(model_name),
            sentences=np.array(sentences, dtype=object),
            embeddings=embeddings,
        )

    # ------------------------------------------------------------------

    def __load_model__(self, model_meta):
        import torch
        model_name = model_meta["model"]

        if model_name not in self._model_cache:
            print(f"[INFO] Loading model {model_name} into cache...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Pass device here: sentence-transformers 5.x loads directly on the
            # target device, avoiding the meta-tensor → .to(device) error path.
            self._model_cache[model_name] = SentenceTransformer(
                model_name,
                cache_folder=f'/Features/Semantic/weights_cache/{model_name}',
                device=device,
            )
        return self._model_cache[model_name]

    def __local__(self):
        for model_meta in tqdm(self.feature_model_local_list, desc="Encoding with models"):
            model_name = model_meta["model"]

            # Load persistent cache for this model (no-op after first call)
            SemanticFeatures.load_embedding_cache(model_name)

            model = self.__load_model__(model_meta)
            sent_cache = self._embedding_cache.setdefault(model_name, {})
            new_sentences = [s for s in self.text_list if s not in sent_cache]

            if new_sentences:
                new_embeddings = model.encode(
                    new_sentences,
                    task=model_meta['task'],
                    prompt=model_meta['prompt']
                )
                for s, emb in zip(new_sentences, new_embeddings):
                    sent_cache[s] = emb

                # Persist after each batch of new sentences
                SemanticFeatures.save_embedding_cache(model_name)

            self.features_dict[model_name] = [sent_cache[s] for s in self.text_list]

    def __api__(self):
        raise NotImplementedError('This Feature is not yet created')

    def run(self):
        self.__local__()


if __name__ == '__main__':
    sample_text = [
        'My Name is Mehul',
        'Mehul is a good person'
    ]

    SampleObj = SemanticFeatures(sample_text)
    SampleObj.run()

    print(SampleObj.features_dict)
