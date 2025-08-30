from Postprocess.postprocess import pad_matrix
from gliner import GLiNER
from collections import Counter
from tqdm import tqdm


class EntityMatch:
    def __init__(self
                 , entity_list=None
                 , MODEL="knowledgator/modern-gliner-bi-base-v1.0"
                 , THRESHOLD=0.3
                 ):
        self.phrase1_list = None
        self.phrase1_entity_cnts = None
        self.phrase2_list = None
        self.phrase2_entity_cnts = None

        self.MODEL = MODEL
        self.THRESHOLD = THRESHOLD
        self.comparison_weights = {'EntityMismatch': []}

        # Set default entity types if none are provided
        default_entity = ["person", "organization", "location", "date", "event", "art", "work_of_art", "law", "language"]
        self.entity_list = entity_list or default_entity

    def __load_model__(self):
        """
        Load the GLiNER model for Named Entity Recognition
        :return: None
        """
        self.__model_cache__ = GLiNER.from_pretrained(
            self.MODEL,
            cache_dir=f"./model/{self.MODEL}/",
        )

    def __get_entities__(self, text):
        """
        Extract entities from the provided text using the GLiNER model.
        :param text: Input text from which to extract entities.
        :return: List of extracted entities.
        """
        if not hasattr(self, '__model_cache__'):
            self.__load_model__()

        entities_val = self.__model_cache__.predict_entities(text, self.entity_list, threshold=self.THRESHOLD)
        entities_dict = Counter([ent['label'] for ent in entities_val])
        return {ent_tag: entities_dict.get(ent_tag, 0) for ent_tag in self.entity_list}

    def __compute_phrase_entity__(self):
        """
        Compute and store entities for both phrase lists.
        :return: None
        """
        self.phrase1_entity_cnts = [self.__get_entities__(text) for text in self.phrase1_list]
        self.phrase2_entity_cnts = [self.__get_entities__(text) for text in self.phrase2_list]

        for dict1 in tqdm(self.phrase1_entity_cnts):
            row = []
            for dict2 in self.phrase2_entity_cnts:
                # Calculate difference in entity counts
                row.append(sum(-1*abs(dict1.get(ent, 0) - dict2.get(ent, 0)) for ent in self.entity_list))
            self.comparison_weights['EntityMismatch'].append(row)

        self.comparison_weights['EntityMismatch'] = pad_matrix(self.comparison_weights['EntityMismatch'])

    def getFeatureMap(self, phrase1_list, phrase2_list):
        """
        Compare two lists of phrases and compute overlap based on entity counts.
        :param phrase1_list: List of phrases from the first source.
        :param phrase2_list: List of phrases from the second source.
        :return: Padded matrix of comparison weights.
        """
        self.__init__()
        self.phrase1_list, self.phrase2_list = phrase1_list, phrase2_list
        self.__compute_phrase_entity__()
        return self.comparison_weights


if __name__ == '__main__':
    sample_text = [
        'My Name is Mehul',
        'Mehul is a good person'
    ]

    SampleObj = EntityMatch()
    print(SampleObj.getFeatureMap(sample_text, sample_text))

    sample_text = [
        'My Name is Mehul',
        'Mehul is a good person',
        'Mehul is sometimes good person, sometimes bad person'
    ]

    print(SampleObj.getFeatureMap(sample_text, sample_text))