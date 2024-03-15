import os
import argparse
import pathlib

import matplotlib.pyplot as plt
from sklearn.metrics import (
    PrecisionRecallDisplay,
)

from chatprotect.util import read_sent_file, read_entities, prompt_identifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dir", type=str, default="output/test/chatgpt/chatgpt/s3/test_m3"
    )
    parser.add_argument("--pr_curve_file", type=str, default=None)
    parser.add_argument("--tikz_plot", action="store_true")
    parser.add_argument("--prf1", action="store_true")
    parser.add_argument("--total", action="store_true")
    parser.add_argument("--convo", action="store_true")
    # these settings are needed for old test sets
    parser.add_argument("--old", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--sent-mode", type=int, default=3)
    parser.add_argument(
        "--test_sentence_dir", type=str, default="test/test/sentences/chatgpt/"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["any", "compactie", "native"],
        default="compactie",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["contradiction", "factuality"],
        default="contradiction",
    )
    # should the label from the sentence file override the label in the output
    parser.add_argument("--override_sent", action="store_true")
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
    test_dir = args.test_dir
    if args.old:
        method = args.method
        entities = read_entities("test/test/entities.txt")
        entities_rev_map = {
            e.lower().replace(" ", "_"): prompt_identifier(f"Please tell me about {e}")
            for e in entities
        }
        test_sentence_dir = pathlib.Path(args.test_sentence_dir)
        sent_map = {}
        for entity_dir in test_sentence_dir.iterdir():
            sent_subdir = entity_dir / f"m{args.sent_mode}"
            for sent_file in sent_subdir.iterdir():
                try:
                    sent_map[str(sent_file)] = read_sent_file(None, sent_file)
                except ValueError:
                    continue

    cost, scores, labels, files, convos = 0, [], [], [], []
    for ent_path in sorted(
        os.listdir(test_dir)
        if not args.parallel
        else os.listdir(os.path.join(test_dir, "1"))
    ):
        if any(ent_path.startswith(s) for s in ("m4", "m3", "m2")):
            continue
        local_scores, local_labels, local_files, local_convos = [], [], [], []
        local_convo = ""
        with open(
            os.path.join(test_dir, ent_path)
            if not args.parallel
            else os.path.join(test_dir, "1", ent_path, "stdout")
        ) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("score:"):
                local_scores.append(float(line[line.find(" ") + 1 :]))
            elif line.startswith("label:"):
                local_convos.append(local_convo)
                local_convo = ""
                label = line[line.find(" ") + 1 :]
                if label == "ok":
                    local_labels.append(0)
                else:
                    local_labels.append(1)
            elif "test/test" in line:
                local_files.append(line)
            else:
                local_convo += line + "\n"

        assert len(local_scores) == len(local_labels)
        if args.old:
            assert len(local_scores) == len(local_files)

        scores.extend(local_scores)
        labels.extend(local_labels)
        files.extend(local_files)
        convos.extend(local_convos)

    assert len(scores) == len(labels)
    if args.old:
        assert len(scores) == len(files)

    TP, FP, TN, FN = 0, 1, 2, 3
    inconsistency_mat = [
        0,
        0,
        0,
        0,
    ]  # TP (strong), FP (ok), TN (ok), FN (strong)
    for i, (label, score, convo) in enumerate(zip(labels, scores, convos)):
        if args.old:
            sent_file = files[i]
            if pathlib.Path(sent_file).stem.startswith("m"):
                continue
            for rev_e, e in entities_rev_map.items():
                sent_file = sent_file.replace(rev_e, e)
            ext_sent = sent_map[sent_file]
            if args.sent_mode == 3 and ext_sent.triple[1] != method:
                continue
            if args.override_sent:
                if args.mode == "contradiction":
                    label = int(ext_sent.tag in ("strong", "weak"))
                else:
                    label = int(ext_sent.wrong in ("both", "1"))

        derived_label = int(score > 0.5)
        if derived_label == 1 and label == 1:
            inconsistency_mat[TP] += 1
        if derived_label == 1 and label == 0:
            if args.convo:
                print("FP")
                print(f"score: {score}")
                print(f"label: {label}")
                print(convo)
            inconsistency_mat[FP] += 1
        if derived_label == 0 and label == 0:
            inconsistency_mat[TN] += 1
        if derived_label == 0 and label == 1:
            if args.convo:
                print("FN")
                print(f"score: {score}")
                print(f"label: {label}")
                print(convo)
            inconsistency_mat[FN] += 1

    if args.total:
        print("TP (strong), FP (ok), TN (ok), FN (strong)")
        print(",".join(map(str, inconsistency_mat)))

    if args.prf1:
        # print(inconsistency_mat)
        precision = inconsistency_mat[TP] / (
            inconsistency_mat[TP] + inconsistency_mat[FP]
        )
        recall = inconsistency_mat[TP] / (inconsistency_mat[TP] + inconsistency_mat[FN])
        f1 = 2 * (precision * recall) / (precision + recall)
        print("P,R,F1")
        print(f"{precision:.3},{recall:.3},{f1:.3}")
    pr = PrecisionRecallDisplay.from_predictions(labels, scores)
    if args.tikz_plot:
        to_tikz_plot(pr)
    if args.pr_curve_file is not None:
        print(pr.average_precision)
        pr.plot()
        plt.savefig(args.pr_curve_file)


if __name__ == "__main__":
    main()
