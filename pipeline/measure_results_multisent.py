import json
import os
import argparse
import pathlib
from collections import defaultdict
from typing import Dict, Set, List

import matplotlib.pyplot as plt
from sklearn.metrics import (
    PrecisionRecallDisplay,
)

from chatprotect.util import read_sent_file, ExtSentence


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_sentence_dir", type=str, default="test/test/sentences/chatgpt/"
    )
    parser.add_argument("--pr_curve_file", type=str, default=None)
    parser.add_argument("--sent-mode", type=int, default=3)
    args = parser.parse_args()
    return args


max_wrong = 30


def main():
    args = get_args()
    # parse sentences
    test_sentence_dir = pathlib.Path(args.test_sentence_dir)
    for mode in ["prec", "rec", "f1", "total"]:
        for test_sentence_entity_dir in test_sentence_dir.iterdir():
            sents = defaultdict(list)
            if not test_sentence_entity_dir.is_dir():
                continue
            for test_file in sorted(
                (test_sentence_entity_dir / f"m{args.sent_mode}").iterdir()
            ):
                try:
                    ext_sent = read_sent_file(None, test_file)
                except ValueError:
                    continue
                sents[(ext_sent.orig, tuple(ext_sent.triple[0]))].append(ext_sent)
            wrong_sent: Set[str] = set(
                e.orig for es in sents.values() for e in es if e.wrong in ("1", "both")
            )
            contra_sent: List[Set[str]] = [set() for _ in range(max_wrong)]

            contradictions = [0] * max_wrong
            for sent, alts in sents.items():
                contra = 0
                for i in range(max_wrong):
                    ext_sent: ExtSentence = alts[i]
                    contra += ext_sent.tag == "strong"
                    contradictions[i] += contra
                    if contra > 0:
                        contra_sent[i].add(ext_sent.orig)
            prec = [
                (len(cs.intersection(wrong_sent)) / len(cs)) if len(cs) else None
                for cs in contra_sent
            ]
            recall = [
                (len(cs.intersection(wrong_sent)) / len(wrong_sent))
                if len(wrong_sent)
                else None
                for cs in contra_sent
            ]
            f1 = [
                2 * (p * r) / (p + r)
                if p is not None and r is not None and (p + r) != 0
                else None
                for p, r in zip(prec, recall)
            ]
            # precision: how many of the contradictory sentences are wrong?
            if mode == "total":
                print(
                    mode
                    + ","
                    + test_sentence_entity_dir.name
                    + ","
                    + (",".join(str(len(cs)) for cs in contra_sent))
                )
            if mode == "prec":
                print(
                    mode
                    + ","
                    + test_sentence_entity_dir.name
                    + ","
                    + (",".join(str(p) for p in prec))
                )
            # recall: how many of the wrong sentences are contradictory?
            if mode == "rec":
                print(
                    mode
                    + ","
                    + test_sentence_entity_dir.name
                    + ","
                    + (",".join(str(p) for p in recall))
                )
            # f1
            if mode == "f1":
                print(
                    mode
                    + ","
                    + test_sentence_entity_dir.name
                    + ","
                    + (",".join(str(p) for p in f1))
                )


if __name__ == "__main__":
    main()
