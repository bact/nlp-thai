import glob
import sys
from argparse import ArgumentParser
from typing import List

import pycrfsuite

import crf_wordseg_util

NGRAM = 11

tagger = pycrfsuite.Tagger()


# Tokenize function
def tokenize(text: str) -> List[str]:
    text_features = crf_wordseg_util.extract_features(text)
    delimiters = tagger.tag(text_features)

    tokens = []
    token = ""
    for i, c in enumerate(text):
        if delimiters[i] == "B":
            if token:
                tokens.append(token)
            token = ""
        token = token + c
    if token:
        tokens.append(token)

    return tokens


# Execute only if run as a script
# Usage:
# crf-wordseg.py wordseg.model -s "ทดสอบ"
# crf-wordseg.py wordseg.model -f data/nectec-best/TEST_100K.txt
if __name__ == "__main__":
    aparser = ArgumentParser(description="Train and segment (tokenize) text")
    aparser.add_argument(
        "-s", "--segment", nargs=2, metavar=("model_filename", "input_text")
    )
    aparser.add_argument(
        "-ss", "--segmentfile", nargs=2, metavar=("model_filename", "input_filename")
    )
    args = vars(aparser.parse_args())

    if len(sys.argv) < 2:
        aparser.print_help()

    if args["segment"]:
        model_filename = args["segment"][0]
        text = args["segment"][1]
        tagger.open(model_filename)
        tokens = tokenize(text)
        print(tokens)
        print("|".join(tokens))

    if args["segmentfile"]:
        model_filename = args["segmentfile"][0]
        input_filename = args["segmentfile"][1]
        tagger.open(model_filename)
        filenames = glob.glob(input_filename)
        for filename in filenames:
            text = open(filename).read()
            tokens = tokenize(text)
            print("|".join(tokens))
