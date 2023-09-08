import json
import os
import argparse
import pathlib

import matplotlib.pyplot as plt
from sklearn.metrics import (
    PrecisionRecallDisplay,
)

from chatprotect.util import read_sent_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_sentence_dir", type=str, default="test/test/sentences/chatgpt/"
    )
    parser.add_argument("--pr_curve_file", type=str, default=None)
    parser.add_argument("--tikz_plot", action="store_true")
    parser.add_argument("--prf1", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        choices=["any", "compactie", "native"],
        default="compactie",
    )
    parser.add_argument("--sent-mode", type=int, default=3)
    args = parser.parse_args()
    return args


def to_tikz_plot(curve):
    last_x, last_y = -1e3, -1e3
    for i, (x, y) in enumerate(zip(curve.line_._x, curve.line_._y)):
        if abs(last_x - x) < 1e-3 and 0.01 < x < 0.999:
            continue
        if abs(last_y - y) < 1e-3 and 0.01 < x < 0.999:
            continue
        last_x, last_y = x, y
        print("        ({}, {})".format(x * 100, y * 100))


def main():
    args = get_args()
    # parse sentences
    sents = {}
    test_sentence_dir = pathlib.Path(args.test_sentence_dir)
    method = args.method
    scores, labels, files = [], [], []
    for test_sentence_entity_dir in test_sentence_dir.iterdir():
        local_scores, local_labels, local_sents = [], [], []
        if not test_sentence_entity_dir.is_dir():
            continue
        for test_file in (test_sentence_entity_dir / f"m{args.sent_mode}").iterdir():
            try:
                ext_sent = read_sent_file(None, test_file)
            except ValueError:
                continue
            ent_path = test_file
            local_scores.append(
                int(
                    (
                        ext_sent.orig_tag
                        if ext_sent.orig_tag is not None
                        else ext_sent.tag
                    )
                    == "strong"
                )
            )
            local_labels.append(int(ext_sent.tag == "strong"))
            local_sents.append(ext_sent)

        assert len(local_scores) == len(local_labels) == len(local_sents)

        TP, FP, TN, FN = 0, 1, 2, 3
        inconsistency_mat = [
            0,
            0,
            0,
            0,
        ]  # TP (strong), FP (ok), TN (ok), FN (strong)
        for label, score, file in zip(local_labels, local_scores, local_sents):
            if args.sent_mode == 3 and method != "any":
                t_method = file.triple[1]
                if t_method != method:
                    continue
            derived_label = int(score > 0.5)
            if derived_label == 1 and label == 1:
                inconsistency_mat[TP] += 1
            elif derived_label == 1 and label == 0:
                inconsistency_mat[FP] += 1
            elif derived_label == 0 and label == 0:
                inconsistency_mat[TN] += 1
            elif derived_label == 0 and label == 1:
                inconsistency_mat[FN] += 1
            else:
                raise NotImplementedError

        print(*inconsistency_mat)

        scores.extend(local_scores)
        labels.extend(local_labels)
        files.extend(local_sents)

    assert len(scores) == len(labels)

    TP, FP, TN, FN = 0, 1, 2, 3
    inconsistency_mat = [
        0,
        0,
        0,
        0,
    ]  # TP (strong), FP (ok), TN (ok), FN (strong)
    for label, score, file in zip(labels, scores, files):
        if args.sent_mode == 3 and method != "any":
            t_method = file.triple[1]
            if t_method != method:
                continue
        derived_label = int(score > 0.5)
        if derived_label == 1 and label == 1:
            inconsistency_mat[TP] += 1
        if derived_label == 1 and label == 0:
            inconsistency_mat[FP] += 1
        if derived_label == 0 and label == 0:
            inconsistency_mat[TN] += 1
        if derived_label == 0 and label == 1:
            inconsistency_mat[FN] += 1

    print(",".join(map(str, inconsistency_mat)))

    pr = PrecisionRecallDisplay.from_predictions(labels, scores)
    if args.tikz_plot:
        to_tikz_plot(pr)
    if args.prf1:
        # print(inconsistency_mat)
        precision = inconsistency_mat[TP] / (
            inconsistency_mat[TP] + inconsistency_mat[FP]
        )
        recall = inconsistency_mat[TP] / (inconsistency_mat[TP] + inconsistency_mat[FN])
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"P:{precision:.3},R:{recall:.3},F1:{f1:.3}")
    if args.pr_curve_file is not None:
        print(pr.average_precision)
        pr.plot()
        plt.savefig(args.pr_curve_file)


if __name__ == "__main__":
    main()
