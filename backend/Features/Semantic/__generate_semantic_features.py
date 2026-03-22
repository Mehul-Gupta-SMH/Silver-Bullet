from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm


class SemanticFeatures:
    # Global cache (shared across all instances)
    _model_cache = {}

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

    def __load_model__(self, model_meta):
        model_name = model_meta["model"]

        if model_name not in self._model_cache:
            # Load once and keep in memory
            print(f"[INFO] Loading model {model_name} into cache...")
            self._model_cache[model_name] = SentenceTransformer(
                model_name,
                cache_folder=f'/Features/Semantic/weights_cache/{model_name}'
            )
        return self._model_cache[model_name]

    def __local__(self):
        for model_meta in tqdm(self.feature_model_local_list, desc="Encoding with models"):
            model = self.__load_model__(model_meta)
            self.features_dict[model_meta["model"]] = model.encode(
                self.text_list,
                task=model_meta['task'],
                prompt=model_meta['prompt']
            )

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
