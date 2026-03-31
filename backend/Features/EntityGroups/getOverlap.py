from backend.Postprocess.__addpad import resize_matrix
from gliner import GLiNER
from collections import Counter
from tqdm import tqdm


class EntityMatch:
    # Shared across all instances — GLiNER loaded once per process
    _model_cache: dict = {}

    def __init__(
        self,
        entity_list=None,
        MODEL="knowledgator/modern-gliner-bi-base-v1.0",
        THRESHOLD=0.3,
    ):
        self.MODEL = MODEL
        self.THRESHOLD = THRESHOLD
        default_entity = [
            "person", "organization", "location", "date", "event",
            "art", "work_of_art", "law", "language"
        ]
        self.entity_list = entity_list or default_entity
        self._reset_state()

    def _reset_state(self):
        """Reset per-call buffers. Preserves the loaded GLiNER model cache."""
        self.phrase1_list = None
        self.phrase1_entity_cnts = None
        self.phrase2_list = None
        self.phrase2_entity_cnts = None
        self.comparison_weights = {'EntityMismatch': []}

    def __load_model__(self):
        EntityMatch._model_cache[self.MODEL] = GLiNER.from_pretrained(
            self.MODEL,
            cache_dir=f"Features/EntityGroups/model/{self.MODEL}/",
        )

    def _batch_get_entities(self, texts):
        """Encode all texts in a single GLiNER call and return a list of entity-count dicts."""
        if self.MODEL not in EntityMatch._model_cache:
            self.__load_model__()

        model = EntityMatch._model_cache[self.MODEL]
        if hasattr(model, 'run'):
            all_results = model.run(
                texts, self.entity_list, threshold=self.THRESHOLD
            )
        elif hasattr(model, 'batch_predict_entities'):
            all_results = model.batch_predict_entities(
                texts, self.entity_list, threshold=self.THRESHOLD
            )
        else:
            all_results = [
                model.predict_entities(t, self.entity_list, threshold=self.THRESHOLD)
                for t in texts
            ]

        out = []
        for entities_val in all_results:
            counts = Counter([ent['label'] for ent in entities_val])
            out.append({tag: counts.get(tag, 0) for tag in self.entity_list})
        return out

    def __compute_phrase_entity__(self):
        all_texts = self.phrase1_list + self.phrase2_list
        all_counts = self._batch_get_entities(all_texts)
        n1 = len(self.phrase1_list)
        self.phrase1_entity_cnts = all_counts[:n1]
        self.phrase2_entity_cnts = all_counts[n1:]

        for dict1 in tqdm(self.phrase1_entity_cnts):
            row = [
                sum(-abs(dict1.get(e, 0) - dict2.get(e, 0)) for e in self.entity_list)
                for dict2 in self.phrase2_entity_cnts
            ]
            self.comparison_weights['EntityMismatch'].append(row)

        self.comparison_weights['EntityMismatch'] = resize_matrix(
            self.comparison_weights['EntityMismatch']
        )

    def getFeatureMap(self, phrase1_list, phrase2_list):
        self._reset_state()
        self.phrase1_list, self.phrase2_list = phrase1_list, phrase2_list
        self.__compute_phrase_entity__()
        return self.comparison_weights


if __name__ == '__main__':
    sample_text = [
        'My Name is Mehul',
        'Mehul is a good person'
    ]
    obj = EntityMatch()
    print(obj.getFeatureMap(sample_text, sample_text))

    sample_text2 = [
        'My Name is Mehul',
        'Mehul is a good person',
        'Mehul is sometimes good person, sometimes bad person'
    ]
    print(obj.getFeatureMap(sample_text2, sample_text2))
