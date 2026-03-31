#################################
# Simple Splitter : Split on sentences
#################################
import re
from backend.Preprocess.coref.resolveEntity import EntityResolver

# Heuristic: if the text looks like code (contains common code tokens) treat
# each logical block (paragraph / blank-line boundary) as one unit rather than
# splitting on every newline or period inside the code.
_CODE_TOKENS = re.compile(
    r'\bdef |class |import |return |if |else:|elif |for |while |try:|except|'
    r'```|#include|public |private |void |int |func |fn |lambda '
)


def _is_code(text: str) -> bool:
    """Return True if the text looks like a code snippet."""
    return bool(_CODE_TOKENS.search(text))


def split_txt(base_text: str, resolver: EntityResolver = None,
              resolve_coref: bool = False, **kwargs) -> list:
    """Split text at sentence boundaries and optionally resolve coreferences.

    For natural-language text: splits on sentence-ending periods and single
    newlines (original behaviour).

    For code-like text: splits on blank lines only, preserving multi-line
    constructs (functions, loops) as single units.

    Args:
        base_text (str): The text to split.
        resolver (EntityResolver | None): Pre-instantiated resolver to reuse.
            Ignored when resolve_coref=False.
        resolve_coref (bool): Whether to run coreference resolution on each
            sentence. Defaults to False. Has no effect on code-like text.
    Returns:
        List[str]: Cleaned sentences / blocks.
    """
    if _is_code(base_text):
        # Split only on blank lines; keep multi-line blocks intact.
        raw = re.split(r'\n\s*\n', base_text)
        return [s.strip() for s in raw if s.strip()]

    # Natural-language path: split on period (not between digits) or newline.
    regex_pattern = r"(?<!\d)\.(?!\d)|\n"
    raw_sentences = re.split(regex_pattern, base_text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not resolve_coref:
        return sentences

    if resolver is None:
        resolver = EntityResolver(kwargs.get('COREF_MODEL', 'gpt-4o-mini'))
    return [resolver.resolve(s) for s in sentences]
