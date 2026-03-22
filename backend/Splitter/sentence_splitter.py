#################################
# Simple Splitter : Split on sentences
#################################
import re
from backend.Preprocess.coref.resolveEntity import EntityResolver


def split_txt(base_text: str, resolver: EntityResolver = None,
              resolve_coref: bool = False, **kwargs) -> list:
    """Split text at sentence boundaries and optionally resolve coreferences.

    Args:
        base_text (str): The text to split.
        resolver (EntityResolver | None): Pre-instantiated resolver to reuse.
            Ignored when resolve_coref=False.
        resolve_coref (bool): Whether to run coreference resolution on each
            sentence. Defaults to False — set True only at inference time when
            an OpenAI key is available. Always leave False during training /
            feature precomputation to avoid API calls.
    Returns:
        List[str]: Cleaned sentences, coref-resolved if resolve_coref=True.
    """
    regex_pattern = r"(?<!\d)\.(?!\d)|\n"
    raw_sentences = re.split(regex_pattern, base_text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not resolve_coref:
        return sentences

    if resolver is None:
        resolver = EntityResolver(kwargs.get('COREF_MODEL', 'gpt-4o-mini'))
    return [resolver.resolve(s) for s in sentences]
