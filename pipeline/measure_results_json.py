import json
import os
import argparse
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.metrics import (
    PrecisionRecallDisplay,
)

from chatprotect.util import read_sent_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dir", type=str, default="output/test/chatgpt/chatgpt/s3/test_m3"
    )
    parser.add_argument("--pr_curve_file", type=str, default=None)
    parser.add_argument("--tikz_plot", action="store_true")
    parser.add_argument("--prf1", action="store_true")
    parser.add_argument("--total", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--convo", action="store_true")
    parser.add_argument("--override_sent", action="store_true")
    parser.add_argument("--sent-mode", type=int, default=3)
    parser.add_argument(
        "--mode", choices=["factuality", "contradiction"], default="contradiction"
    )
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


sent_map = defaultdict(list)
for ent_dir in pathlib.Path("test/test/sentences/chatgpt/").iterdir():
    for sent_file in (ent_dir / "m3").iterdir():
        try:
            orig_ext_sent = read_sent_file(None, sent_file)
        except:
            continue
        sent_map[orig_ext_sent.orig].append(orig_ext_sent)


def print_stats(labels, scores, files, convos, args):
    TP, FP, TN, FN = 0, 1, 2, 3
    inconsistency_mat = [
        0,
        0,
        0,
        0,
    ]  # TP (strong), FP (ok), TN (ok), FN (strong)
    for label, score, file, convo in zip(labels, scores, files, convos):
        try:
            orig_ext_sent = read_sent_file(None, file)
        except:
            continue
        if orig_ext_sent.triple[1] != "compactie":
            continue
        if args.override_sent:
            # override label with most recent annotation
            label = 1 if orig_ext_sent.tag in ("strong", "weak") else 0
        derived_label = int(score > 0.5)
        if derived_label == 1 and label == 1:
            inconsistency_mat[TP] += 1
        if derived_label == 1 and label == 0:
            if args.convo:
                print("FP")
                print(f"score: {score}")
                print(f"label: {label}")
                print(convo[0]["Q"])
                print(convo[0]["A"][0])
                print()
                print()
                print("=" * 100)
            inconsistency_mat[FP] += 1
        if derived_label == 0 and label == 0:
            inconsistency_mat[TN] += 1
        if derived_label == 0 and label == 1:
            if args.convo:
                print("FN")
                print(f"score: {score}")
                print(f"label: {label}")
                print(convo[0]["Q"])
                print(convo[0]["A"][0])
                print()
                print()
                print("=" * 100)
            inconsistency_mat[FN] += 1

    if args.total:
        print("TP (strong), FP (ok), TN (ok), FN (strong)")
        print(",".join(map(str, inconsistency_mat)))

    pr = PrecisionRecallDisplay.from_predictions(labels, scores)
    print(pr.average_precision)
    if args.tikz_plot:
        to_tikz_plot(pr)
    if args.prf1:
        # print(inconsistency_mat)
        precision = (
            (inconsistency_mat[TP] / (inconsistency_mat[TP] + inconsistency_mat[FP]))
            if inconsistency_mat[TP] + inconsistency_mat[FP]
            else 1
        )
        recall = (
            (inconsistency_mat[TP] / (inconsistency_mat[TP] + inconsistency_mat[FN]))
            if inconsistency_mat[TP] + inconsistency_mat[FN]
            else 1
        )
        f1 = (
            (2 * (precision * recall) / (precision + recall))
            if precision + recall
            else 1
        )
        print("P,R,F1")
        print(f"{precision:.3f},{recall:.3f},{f1:.3f}")
    return pr


def main():
    args = get_args()
    # parse sentences
    test_dir = args.test_dir
    mode = args.mode
    cost, scores, labels, files, convos = 0, [], [], [], []
    for ent_path in sorted(os.listdir(test_dir)):
        if any(ent_path.startswith(s) for s in ("m4", "m3", "m2")):
            continue
        local_scores, local_labels, local_files, local_convos = [], [], [], []
        local_convo = []
        with open(os.path.join(test_dir, ent_path)) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                d = json.loads(line)
            except:
                continue
            if "Q" in d:
                local_convo.append(d)
            if "score" not in d:
                continue
            convo = local_convo
            local_convo = []
            score = d.get("score")
            if mode == "factuality":
                label = 1 if d["wrong"] in ("1", "both") else 0
            elif mode == "contradiction":
                label = 1 if d["label"] == "strong" else 0
            else:
                raise NotImplementedError()
            file = d.get("file")
            if args.override_sent:
                if args.mode == "contradiction":
                    try:
                        orig_ext_sent = read_sent_file(None, file)
                    except:
                        continue
                    if orig_ext_sent.triple[1] != "compactie":
                        continue
                    # override label with most recent annotation
                    label = 1 if orig_ext_sent.tag in ("strong", "weak") else 0
                else:
                    orig_sents = sent_map[d.get("orig")]
                    if any(s.wrong in ("1", "both") for s in orig_sents):
                        label = 1
            local_labels.append(label)
            local_convos.append(convo)
            local_scores.append(score)
            local_files.append(file)

        assert len(local_scores) == len(local_labels)

        if args.local:
            name = pathlib.Path(ent_path).stem
            print(name)
            pr = print_stats(
                local_labels, local_scores, local_files, local_convos, args
            )
            if args.pr_curve_file is not None:
                pr.plot()
                plt.savefig(args.pr_curve_file.format(name))
        scores.extend(local_scores)
        labels.extend(local_labels)
        files.extend(local_files)
        convos.extend(local_convos)

    assert len(scores) == len(labels)

    if not args.local:
        pr = print_stats(labels, scores, files, convos, args)
        if args.pr_curve_file is not None:
            pr.plot()
            plt.savefig(args.pr_curve_file)


if __name__ == "__main__":
    main()
