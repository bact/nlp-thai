import glob
import time
from argparse import ArgumentParser
from typing import List, Tuple

import crf_wordseg_util


# Take text in BEST corpus format, return a list of (character, label) tuples
def best_corpus_text_to_tuples(text: str) -> List[Tuple[str, str]]:
    text = text.replace("<NE>", "")
    text = text.replace("</NE>", "")
    text = text.replace("\n", "")

    tuples = []
    tokens = text.split("|")
    for token in tokens:
        for i in range(len(token)):
            label = ""
            if i == 0:
                label = "B"  # Beginning of word
            else:
                label = "I"  # Inside word
            tuples.append((token[i], label))

    return tuples


# Get only labels (use in evaluation)
def get_labels(char_label_tuples) -> List[str]:
    return [label for (char, label) in char_label_tuples]


# Make training data
def make_train(input_filename: str, train_filename: str, ngram: int, dict_automaton):
    print("N-gram:", ngram)
    print("Dictionary words:", len(dict_automaton))
    print("Character types:")
    for chartype in crf_wordseg_util.CHARTYPES:
        print("Type {}: {}".format(chartype[0], chartype[1]))

    filenames = glob.glob(input_filename)
    filenames.sort()
    len_filenames = len(filenames)
    print("Training input files: {}".format(len_filenames))
    if len_filenames < 6:
        print(", ".join(filenames))
    else:
        print(", ".join(filenames[0:4]) + " ... " + filenames[-1])
    with open(train_filename, "w") as train_filename:
        for filename in filenames:
            with open(filename, "r") as input_file:
                best_corpus_text = input_file.read()
                labels = get_labels(best_corpus_text_to_tuples(best_corpus_text))
                text = crf_wordseg_util.remove_best_corpus_labels(best_corpus_text)
                features = crf_wordseg_util.extract_features(text, ngram)
                for char_label, char_features in zip(labels, features):
                    train_filename.write("{}\t".format(char_label))
                    for char_feature in char_features:
                        train_filename.write("{}\t".format(char_feature))
                    train_filename.write("\n")
            print(".", end="")
    print("\nDone.")


# Execute only if run as a script
# Usage:
# crf_wordseg_make_train.py -n 13 "data/nectec-best/*/*.txt"
if __name__ == "__main__":
    aparser = ArgumentParser(description="Make training data for word tokenizer")
    aparser.add_argument("-n", "--ngram")
    aparser.add_argument("-d", "--dictionary")
    aparser.add_argument("input_filename")
    aparser.add_argument("train_filename")
    args = vars(aparser.parse_args())

    ngram = crf_wordseg_util.DEFAULT_NGRAM
    if args["ngram"]:
        ngram = int(args["ngram"])

    dict_automaton = crf_wordseg_util.DICT_AUTOMATON
    if args["dictionary"]:
        dict_automaton = crf_wordseg_util.get_dict_automaton(args["dictionary"])

    start_time = time.time()
    make_train(args["input_filename"], args["train_filename"], ngram, dict_automaton)
    end_time = time.time()
    print("Elasped time: {}".format(crf_wordseg_util.elasped_time(start_time, end_time)))
