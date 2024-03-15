import json
import sys

from chatprotect.util import read_sent_file

for temp in [
    "0.00",
    "0.25",
    "0.50",
    "0.75",
]:
    annotations_file = f"annotations/glm/reconciled/c_c_glm{temp}.json"
    with open(annotations_file) as f:
        annotations = json.load(f)

    all = 0
    all_correct = 0
    for annotation in annotations:
        file = annotation["fn"]
        if file == "demo":
            continue
        sent = read_sent_file(None, file)
        gt = "strong" in annotation["label"]
        pred = "strong" == sent.tag
        all += 1
        all_correct += gt == pred
    print(temp, f"{100 * all_correct / all:.1f}")
