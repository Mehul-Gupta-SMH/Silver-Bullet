#################################
# Simple Splitter : Split on sentences
#################################
import re
from Preprocess.coref.resolveEntity import EntityResolver


def split_txt(base_text: str, **kwargs) -> list:
    """
    Splits the input text at every occurrence of a period ('.') or newline ('\n').

    :param base_text: The text to be split.
    :return: A list of substrings obtained after splitting the text.
    """
    split_txt_list = list()
    EntityResolver_Obj = EntityResolver(kwargs.get('COREF_MODEL', 'gpt-4o-mini'))

    regex_patter = r"(?<!\d)\.(?!\d)|\n"

    split_txt_list = re.split(regex_patter, base_text)

    split_txt_list = [
        EntityResolver_Obj.resolve(sentence.strip())
        for sentence in split_txt_list
        if len(sentence.strip())
    ]

    return split_txt_list
