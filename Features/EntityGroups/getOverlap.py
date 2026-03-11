from Postprocess.__addpad import pad_matrix
from gliner import GLiNER
from collections import Counter
from tqdm import tqdm


class EntityMatch:
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
        self.__model_cache__ = GLiNER.from_pretrained(
            self.MODEL,
            cache_dir=f"Features/EntityGroups/model/{self.MODEL}/",
        )

    def __get_entities__(self, text):
        if not hasattr(self, '__model_cache__'):
            self.__load_model__()

        entities_val = self.__model_cache__.predict_entities(
            text, self.entity_list, threshold=self.THRESHOLD
        )
        entities_dict = Counter([ent['label'] for ent in entities_val])
        return {tag: entities_dict.get(tag, 0) for tag in self.entity_list}

    def __compute_phrase_entity__(self):
        self.phrase1_entity_cnts = [self.__get_entities__(t) for t in self.phrase1_list]
        self.phrase2_entity_cnts = [self.__get_entities__(t) for t in self.phrase2_list]

        for dict1 in tqdm(self.phrase1_entity_cnts):
            row = [
                sum(-abs(dict1.get(e, 0) - dict2.get(e, 0)) for e in self.entity_list)
                for dict2 in self.phrase2_entity_cnts
            ]
            self.comparison_weights['EntityMismatch'].append(row)

        self.comparison_weights['EntityMismatch'] = pad_matrix(
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
