import os
import argparse
import pathlib

import matplotlib.pyplot as plt
from sklearn.metrics import (
    PrecisionRecallDisplay,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dir", type=str, default="output/test/chatgpt/chatgpt/s3/test_m5"
    )
    parser.add_argument("--pr_curve_file", type=str, default=None)
    parser.add_argument(
        "--method",
        type=str,
        choices=["any", "compactie", "native"],
        default="compactie",
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


def main():
    args = get_args()
    test_dir = os.path.join(args.test_dir, "1")
    method = args.method
    scores, labels = [], []
    for ent in sorted(os.listdir(test_dir)):
        ent_path = os.path.join(test_dir, ent, "stdout")
        with open(ent_path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("score:"):
                    scores.append(float(line[line.find(" ") + 1 :]))
                elif line.startswith("label:"):
                    label = line[line.find(" ") + 1 :]
                    if label == "ok":
                        labels.append(0)
                    else:
                        labels.append(1)

    assert len(scores) == len(labels)

    pr = PrecisionRecallDisplay.from_predictions(labels, scores)
    # to_tikz_plot(pr)
    print(pr.average_precision)
    if args.pr_curve_file is not None:
        pr.plot()
        plt.savefig(args.pr_curve_file)


if __name__ == "__main__":
    main()
