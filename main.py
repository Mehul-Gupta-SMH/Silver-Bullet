from Splitter.sentence_splitter import split_txt

from Features.Semantic.getSemanticWeights import SemanticWeights
from Features.Lexical.getLexicalWeights import LexicalWeights

if __name__ == '__main__':
    para1 = """Machine learning models are increasingly being used to analyze large datasets and uncover hidden patterns. These models rely on algorithms that learn from past data and improve their predictions over time. With sufficient training, they can detect trends, classify information, and even generate human-like text."""
    para2 = """Algorithms in artificial intelligence help systems make sense of vast amounts of information by identifying patterns and correlations. By training on existing data, these systems gradually refine their accuracy, enabling them to perform tasks such as text generation, classification, and forecasting."""

    sentence_group1 = split_txt(para1)
    sentence_group2 = split_txt(para2)

    LexicalWeights_Obj = LexicalWeights(sentence_group1, sentence_group2)
    print(LexicalWeights_Obj.getFeatureMap())

    SemanticWeights_Obj = SemanticWeights(sentence_group1=sentence_group1, sentence_group2=sentence_group2)
    print(SemanticWeights_Obj.getFeatureMap())

