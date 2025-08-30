from Features.Semantic.__generate_semantic_features import SemanticFeatures
from Postprocess.__addpad import pad_matrix
from sentence_transformers.util import cos_sim
from torch import softmax, tensor, float
from tqdm import tqdm


class SemanticWeights:
    def __init__(self):
        """
        1. Generate Semantic Features for both sentence groups
        2. Compare the features and generate weights
        3. Return the weights as a dictionary
        4. The weights can be used to compare the two sentence groups
        :param sentence_group1: List of sentences in group 1
        :param sentence_group2: List of sentences in group 2
        :return: Dictionary of weights
        """
        self.sentence_group1 = None
        self.sentence_group1_features = None

        self.sentence_group2 = None
        self.sentence_group2_features = None

        self.comparison_weights = {}

    def __generate_sematic_features__(self):
        SemanticFeatures_Obj = SemanticFeatures(self.sentence_group1)
        SemanticFeatures_Obj.run()
        self.sentence_group1_features = SemanticFeatures_Obj.features_dict

        SemanticFeatures_Obj = SemanticFeatures(self.sentence_group2)
        SemanticFeatures_Obj.run()
        self.sentence_group2_features = SemanticFeatures_Obj.features_dict

    def __calc_weights__(self):
        """

        :return:
        """

        feature_mapval_list = self.sentence_group1_features.keys()

        for feature_mapval in feature_mapval_list:
            g1 = self.sentence_group1_features[feature_mapval]
            g2 = self.sentence_group2_features[feature_mapval]

            rows = []
            for feature1 in tqdm(g1):
                row = []
                for feature2 in g2:  # inner tqdm not usually helpful
                    row.append(cos_sim(feature1, feature2).item())
                rows.append(row)

            self.comparison_weights[feature_mapval] = rows

    def __calc_soft_alignment__(self):
        """
        Calculate soft alignment between the two sentence groups
        :return:
        """
        comparison_weights_inter = {}
        for feature_mapval in tqdm(self.comparison_weights.keys()):
            comparison_weights_inter[f'SOFT_ROW_{feature_mapval}'] = softmax(tensor(self.comparison_weights[feature_mapval],
                                                                                   dtype=float), dim=1).tolist()
            comparison_weights_inter[f'SOFT_COL_{feature_mapval}'] = softmax(tensor(self.comparison_weights[feature_mapval],
                                                                                   dtype=float), dim=0).tolist()
        self.comparison_weights.update(comparison_weights_inter)

    def __post_process_weights__(self):
        for key in self.comparison_weights:
            self.comparison_weights[key] = pad_matrix(self.comparison_weights[key])

    def getFeatureMap(self, sentence_group1, sentence_group2):
        self.__init__()
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

    SampleObj = SemanticWeights()
    print(SampleObj.getFeatureMap(sample_text, sample_text))

    sample_text = [
        'My Name is Mehul',
        'Mehul is a good person',
        'Mehul is sometimes good person, sometimes bad person'
    ]
