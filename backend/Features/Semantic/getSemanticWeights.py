from backend.Features.Semantic.__generate_semantic_features import SemanticFeatures
from backend.Postprocess.__addpad import pad_matrix
from sentence_transformers.util import cos_sim
from torch import softmax, tensor, float
from tqdm import tqdm


class SemanticWeights:
    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        """Reset per-call data. Does not affect the shared SemanticFeatures model cache."""
        self.sentence_group1 = None
        self.sentence_group1_features = None
        self.sentence_group2 = None
        self.sentence_group2_features = None
        self.comparison_weights = {}

    def __generate_sematic_features__(self):
        obj1 = SemanticFeatures(self.sentence_group1)
        obj1.run()
        self.sentence_group1_features = obj1.features_dict

        obj2 = SemanticFeatures(self.sentence_group2)
        obj2.run()
        self.sentence_group2_features = obj2.features_dict

    def __calc_weights__(self):
        for feature_key in self.sentence_group1_features.keys():
            g1 = self.sentence_group1_features[feature_key]
            g2 = self.sentence_group2_features[feature_key]

            rows = []
            for feature1 in tqdm(g1):
                row = [cos_sim(feature1, feature2).item() for feature2 in g2]
                rows.append(row)

            self.comparison_weights[feature_key] = rows

    def __calc_soft_alignment__(self):
        inter = {}
        for key in tqdm(self.comparison_weights.keys()):
            t = tensor(self.comparison_weights[key], dtype=float)
            inter[f'SOFT_ROW_{key}'] = softmax(t, dim=1).tolist()
            inter[f'SOFT_COL_{key}'] = softmax(t, dim=0).tolist()
        self.comparison_weights.update(inter)

    def __post_process_weights__(self):
        for key in self.comparison_weights:
            self.comparison_weights[key] = pad_matrix(self.comparison_weights[key])

    def getFeatureMap(self, sentence_group1, sentence_group2):
        self._reset_state()
        self.sentence_group1, self.sentence_group2 = sentence_group1, sentence_group2
        self.__generate_sematic_features__()
        self.__calc_weights__()
        self.__calc_soft_alignment__()
        self.__post_process_weights__()
        return self.comparison_weights


if __name__ == '__main__':
    sample_text = [
        'My Name is Mehul',
        'Mehul is a good person'
    ]
    obj = SemanticWeights()
    print(obj.getFeatureMap(sample_text, sample_text))
