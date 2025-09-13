from Splitter.sentence_splitter import split_txt

from Features.Semantic.getSemanticWeights import SemanticWeights
from Features.Lexical.getLexicalWeights import LexicalWeights
from Features.NLI.getNLIweights import NLIWeights
from Features.EntityGroups.getOverlap import EntityMatch

from tqdm import tqdm

if __name__ == '__main__':
    LexicalWeights_Obj = LexicalWeights()
    SemanticWeights_Obj = SemanticWeights()
    NLIWeights_Obj = NLIWeights()
    EntityMatch_Obj = EntityMatch()

    para_pairs_list = [
        [
            "My Name is Mehul. Mehul is a good person.",
            "My Name is Mehul. Mehul is a good person. But Mehul can be bad sometimes."
        ],
        [
            "My Name is Mehul. Mehul is a good person.",
            "My Name is Mehul. Mehul is a good person. But he can be bad sometimes."
        ],
        [
            "My Name is Mehul. Mehul is a good person.",
            "My Name is Mehul. Mehul is a good person. But he can be bad sometimes."
        ],
    ]  # List to hold pairs of paragraphs

    for pairs in tqdm(para_pairs_list, 'Processing paragraph pairs'):
        combined_feature_map = {}

        sentence_group1 = split_txt(pairs[0])
        sentence_group2 = split_txt(pairs[1])

        combined_feature_map.update(LexicalWeights_Obj.getFeatureMap(sentence_group1, sentence_group2))
        combined_feature_map.update(SemanticWeights_Obj.getFeatureMap(sentence_group1, sentence_group2))
        combined_feature_map.update(NLIWeights_Obj.getFeatureMap(sentence_group1, sentence_group2))
        combined_feature_map.update(EntityMatch_Obj.getFeatureMap(sentence_group1, sentence_group2))

        print(combined_feature_map)


    

