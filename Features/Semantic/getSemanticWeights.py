from Features.Semantic.generate_semantic_features import SemanticFeatures
from sentence_transformers.util import cos_sim
from tqdm import tqdm


class SemanticWeights:
    def __init__(self, sentence_group1, sentence_group2):
        """
        1. Generate Semantic Features for both sentence groups
        2. Compare the features and generate weights
        3. Return the weights as a dictionary
        4. The weights can be used to compare the two sentence groups
        :param sentence_group1: List of sentences in group 1
        :param sentence_group2: List of sentences in group 2
        :return: Dictionary of weights
        """
        self.sentence_group1 = sentence_group1
        self.sentence_group1_features = None

        self.sentence_group2 = sentence_group2
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

    def getFeatureMap(self):
        self.__generate_sematic_features__()
        self.__calc_weights__()
        return self.comparison_weights


if __name__ == '__main__':
    sample_text = [
        'My Name is Mehul',
        'Mehul is a good person'
    ]

    SampleObj = SemanticWeights(sample_text,sample_text)

    print(SampleObj.getFeatureMap())
