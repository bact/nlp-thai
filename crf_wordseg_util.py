from typing import List

import ahocorasick

DEFAULT_NGRAM = 11
DICT_FILENAME = "data/dict-th.txt"


# Build a dictionary automaton from words provided in a file
def get_dict_automaton(filename: str) -> ahocorasick.Automaton:
    dict_data = []
    with open(filename, "r") as dict_stream:
        for word in dict_stream.readlines():
            dict_data.append(word.strip())
    dict_data = list(set(dict_data))

    dict_automaton = ahocorasick.Automaton()
    for word in dict_data:
        # insert word, with length of word
        # this will be used later to calculate start_index and end_index (as features)
        dict_automaton.add_word(word, len(word))
    del dict_data
    dict_automaton.make_automaton()

    return dict_automaton


DICT_AUTOMATON = get_dict_automaton(DICT_FILENAME)


# Define character types. Types adapted from Haruechaiyasak et al. 2008.
# Character that can be the final consonant in a word
CHARTYPE_C = (
    "\u0e01\u0e02\u0e03\u0e04\u0e06\u0e07\u0e08\u0e0a\u0e0b\u0e0d\u0e0e\u0e0f\u0e10"
    + "\u0e11\u0e12\u0e13\u0e14\u0e15\u0e16\u0e17\u0e18\u0e19\u0e1a\u0e1b\u0e1e\u0e1f"
    + "\u0e20\u0e21\u0e22\u0e23\u0e24\u0e25\u0e26\u0e27\u0e28\u0e29\u0e2a\u0e2c\u0e2d"
)

# Character that cannot be the final consonant in a word
CHARTYPE_N = "\u0e05\u0e09\u0e0c\u0e1c\u0e1d\u0e2b\u0e2e"

# Vowel that cannot begin a word
CHARTYPE_V = "\u0e30\u0e31\u0e32\u0e33\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e45\u0e47"

# Vowel that can begin a word
CHARTYPE_W = "\u0e40\u0e41\u0e42\u0e43\u0e44"

# Combining symbol
CHARTYPE_S = "\u0e3a\u0e4c\u0e4d\u0e4e"

# Standalone symbol
CHARTYPE_A = "\u0e2f\u0e46\u0e4f\u0e5a\u0e5b"

# Tone marks
CHARTYPE_T = "\u0e48\u0e49\u0e4a\u0e4b"

# Digit character
CHARTYPE_D = "0123456789\u0e50\u0e51\u0e52\u0e53\u0e54\u0e55\u0e56\u0e57\u0e58\u0e59"

# Currency character
CHARTYPE_B = "$฿"

# Quote character
CHARTYPE_Q = "'\""

# Other character
CHARTYPE_O = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Space character inside a word
# chartype_p

# Space character
CHARTYPE_Z = " \u00a0\n"

# Undefined
# chartype_x

CHARTYPES = [
    ("c", CHARTYPE_C),
    ("n", CHARTYPE_N),
    ("v", CHARTYPE_V),
    ("w", CHARTYPE_W),
    ("s", CHARTYPE_S),
    ("a", CHARTYPE_A),
    ("t", CHARTYPE_T),
    ("d", CHARTYPE_D),
    ("b", CHARTYPE_B),
    ("q", CHARTYPE_Q),
    ("o", CHARTYPE_O),
    ("z", CHARTYPE_Z),
]

# Experimental: For these character types,
# we assume that an individual character will not be important for classification,
# so in feature generation we will only use the type not the actual character.
GENERIC_CHARTYPES = ["d", "b", "q", "o", "z", "x"]


def elasped_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


# Remove BEST corpus annotations
def remove_best_corpus_labels(text: str) -> str:
    text = text.replace("<NE>", "")
    text = text.replace("</NE>", "")
    text = text.replace("\n", "")
    text = text.replace("|", "")

    return text


# Character type
def get_chartype(char: str) -> str:
    for chartype in CHARTYPES:
        if char in chartype[1]:
            return chartype[0]
    return "x"  # For undefined


# Extract features of each character from text, in CRFSuite format
def extract_features(
    doc: str,
    ngram: int = DEFAULT_NGRAM,
    dict_automaton: ahocorasick.Automaton = DICT_AUTOMATON,
) -> List[List]:
    len_doc = len(doc)
    look_range = list(range(1, int(ngram / 2) + 1))

    # Get (start, end) candidates from dictionary
    dict_start_boundaries = set()
    dict_end_boundaries = set()
    for end_index, length in dict_automaton.iter(doc):
        start_index = end_index - length + 1
        dict_start_boundaries.add(start_index)
        dict_end_boundaries.add(end_index)

    doc_features = []
    for i, char in enumerate(doc):
        ct = get_chartype(char)
        char_features = ["bias", "t={}".format(ct)]
        if ct not in GENERIC_CHARTYPES:
            if char == "\n":
                char = "EOL"
            char_features.append("c={}".format(char))

        if i == 0:
            char_features.append("BOS")  # Beginning of string
        elif i == len_doc - 1:
            char_features.append("EOS")  # End of string

        # Look backward
        for j in look_range:
            if i >= j:
                c = doc[i - j]
                ct = get_chartype(c)
                char_features.append("t-{}={}".format(j, ct))
                if ct not in GENERIC_CHARTYPES:
                    if char == "\n":
                        char = "EOL"
                    char_features.append("c-{}={}".format(j, c))
            else:
                break

        # Look forward
        for j in look_range:
            if i < len_doc - j:
                c = doc[i + j]
                ct = get_chartype(c)
                char_features.append("t{}={}".format(j, ct))
                if ct not in GENERIC_CHARTYPES:
                    if char == "\n":
                        char = "EOL"
                    char_features.append("c{}={}".format(j, c))
            else:
                break

        dict_start_boundary = "n"
        if i in dict_start_boundaries:
            dict_start_boundary = "y"
        char_features.append("ds=" + dict_start_boundary)

        dict_end_boundary = "n"
        if i in dict_end_boundaries:
            dict_end_boundary = "y"
        char_features.append("de=" + dict_end_boundary)

        doc_features.append(char_features)

    return doc_features
