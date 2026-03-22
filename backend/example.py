from backend.Splitter.sentence_splitter import split_txt

from backend.Features.Semantic.getSemanticWeights import SemanticWeights
from backend.Features.Lexical.getLexicalWeights import LexicalWeights
from backend.Features.NLI.getNLIweights import NLIWeights
from backend.Features.EntityGroups.getOverlap import EntityMatch
from backend.Features.LCS.getLCSweights import LCSWeights

from tqdm import tqdm

if __name__ == '__main__':
    lexical  = LexicalWeights()
    semantic = SemanticWeights()
    nli      = NLIWeights()
    entity   = EntityMatch()
    lcs      = LCSWeights()

    para_pairs_list = [
        [
            "My Name is Mehul. Mehul is a good person.",
            "My Name is Mehul. Mehul is a good person. But Mehul can be bad sometimes."
        ],
        [
            "My Name is Mehul. Mehul is a good person.",
            "My Name is Mehul. Mehul is a good person. But he can be bad sometimes."
        ],
    ]

    for pairs in tqdm(para_pairs_list, desc='Processing paragraph pairs'):
        sentence_group1 = split_txt(pairs[0])
        sentence_group2 = split_txt(pairs[1])

        combined_feature_map = {}
        combined_feature_map.update(lexical.getFeatureMap(sentence_group1, sentence_group2))
        combined_feature_map.update(semantic.getFeatureMap(sentence_group1, sentence_group2))
        combined_feature_map.update(nli.getFeatureMap(sentence_group1, sentence_group2))
        combined_feature_map.update(entity.getFeatureMap(sentence_group1, sentence_group2))
        combined_feature_map.update(lcs.getFeatureMap(sentence_group1, sentence_group2))

        print(f"\n--- Pair ---")
        for k, v in combined_feature_map.items():
            print(f"  {k}: shape={v.shape}  min={v.min():.3f}  max={v.max():.3f}")
