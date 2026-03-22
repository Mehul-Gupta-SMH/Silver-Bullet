#################################
# Simple Splitter : Split on sentences
#################################
import re
from backend.Preprocess.coref.resolveEntity import EntityResolver


def split_txt(base_text: str, resolver: EntityResolver = None, **kwargs) -> list:
    """Split text at sentence boundaries and optionally resolve coreferences.

    Args:
        base_text (str): The text to split.
        resolver (EntityResolver | None): Pre-instantiated resolver to reuse.
            If None, a new resolver is created using `kwargs['COREF_MODEL']`
            (default 'gpt-4o-mini').  Pass an existing instance to avoid
            re-loading the model on every call — critical during training.
    Returns:
        List[str]: Cleaned, coref-resolved sentences.
    """
    if resolver is None:
        resolver = EntityResolver(kwargs.get('COREF_MODEL', 'gpt-4o-mini'))

    regex_pattern = r"(?<!\d)\.(?!\d)|\n"
    raw_sentences = re.split(regex_pattern, base_text)

    return [
        resolver.resolve(sentence.strip())
        for sentence in raw_sentences
        if len(sentence.strip())
    ]
