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
        "--test_dir", type=str, default="output/test/chatgpt/chatgpt/s3/test_m5"
    )
    parser.add_argument("--pr_curve_file", type=str, default=None)
    parser.add_argument("--to_tikz", type=str, default=None)
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


def main():
    args = get_args()
    test_dir = os.path.join(args.test_dir, "1")
    scores, labels = [], []
    for ent in sorted(os.listdir(test_dir)):
        ent_path = os.path.join(test_dir, ent, "stdout")
        with open(ent_path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                try:
                    line = json.loads(line)
                except Exception:
                    continue
                if "score" not in line:
                    continue
                scores.append(float(line["score"]))
                orig_ext_sent = sent_map[line["orig"]]
                labels.append(
                    float(any(e.wrong in ("both", "1") for e in orig_ext_sent))
                )

    assert len(scores) == len(labels)

    pr = PrecisionRecallDisplay.from_predictions(labels, scores)
    if args.to_tikz:
        to_tikz_plot(pr)
    print(pr.average_precision)
    if args.pr_curve_file is not None:
        pr.plot()
        plt.savefig(args.pr_curve_file)


if __name__ == "__main__":
    main()
