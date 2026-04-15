import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm

from backend.cache_db import CacheDB


class SemanticFeatures:
    # Shared across all instances: loaded SentenceTransformer objects
    _model_cache = {}
    # Shared across all instances: model_name → {sentence: np.ndarray}
    _embedding_cache: dict = {}
    # Guard: set of model names whose SQLite cache has been loaded this process
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
        """Bulk-load all embeddings for *model_name* from SQLite (once per process)."""
        if model_name in cls._disk_loaded:
            return
        cls._disk_loaded.add(model_name)
        try:
            rows = CacheDB.get().load_all_embeddings(model_name)
            sent_cache = cls._embedding_cache.setdefault(model_name, {})
            sent_cache.update(rows)
            print(f"[EmbedCache] loaded {len(rows)} embeddings for {model_name} from SQLite")
        except Exception as ex:
            print(f"[EmbedCache] warning: could not load cache for {model_name} ({ex}) — starting fresh")

    @classmethod
    def save_embedding_cache(cls, model_name: str, new_sentences: list[str]) -> None:
        """Persist newly computed embeddings for *model_name* to SQLite.

        Args:
            model_name: The embedding model identifier.
            new_sentences: Sentences whose embeddings were just computed.
        """
        if not new_sentences:
            return
        sent_cache = cls._embedding_cache.get(model_name, {})
        embeddings = [sent_cache[s] for s in new_sentences if s in sent_cache]
        if not embeddings:
            return
        try:
            CacheDB.get().save_embeddings_batch(model_name, new_sentences, embeddings)
        except Exception as ex:
            print(f"[EmbedCache] warning: could not save embeddings for {model_name} ({ex})")

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
                SemanticFeatures.save_embedding_cache(model_name, new_sentences)

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
