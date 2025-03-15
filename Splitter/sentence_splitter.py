#################################
# Simple Splitter : Split on sentences
#################################
import re


def split_txt(base_text: str) -> list:
    """
    Splits the input text at every occurrence of a period ('.') or newline ('\n').

    :param base_text: The text to be split.
    :return: A list of substrings obtained after splitting the text.
    """
    split_txt_list = list()
    regex_patter = r"(?<!\d)\.(?!\d)|\n"

    split_txt_list = re.split(regex_patter, base_text)

    split_txt_list = [sentence.strip() for sentence in split_txt_list if len(sentence.strip())]

    return split_txt_list
