import json
import sys
from collections import defaultdict
from pathlib import Path

import datasets

from chatprotect.util import read_desc_file, read_sent_file, split_sentences


def main():
    METHOD = "contriever"
    test_super_set = "popqa"
    dataset = datasets.load_dataset("akariasai/PopQA")["test"]
    for model in ("chatgpt", "llama-2-70b-chat"):
        print(model)
        # Load the dataset
        ds_by_prompt = {d["question"]: d.copy() for d in dataset}
        fn_to_prompt = {}
        for method in ("", f"_{METHOD}"):
            test_set = f"{test_super_set}{method}"

            sentences_dir = Path(f"test/{test_set}/sentences/{model}")
            sentences_dict = defaultdict(list)
            for prompt in sentences_dir.iterdir():
                for answer in (prompt / "m4").iterdir():
                    try:
                        ext_sent = read_sent_file(None, answer)
                    except:
                        continue
                    sentences_dict[ext_sent.orig].append(ext_sent)
                    fn_to_prompt[answer.absolute()] = ext_sent.target

            answers_dir = Path(f"test/{test_set}/descriptions/{model}")
            for prompt in answers_dir.iterdir():
                for answer in prompt.iterdir():
                    desc = read_desc_file(None, answer)
                    possible_answers = json.loads(
                        ds_by_prompt[desc.prompt]["possible_answers"]
                    )

                    accurate = False
                    for pa in possible_answers:
                        if (
                            pa in desc.text
                            or pa.lower() in desc.text
                            or pa.capitalize() in desc.text
                        ):
                            accurate = True
                    ds_by_prompt[desc.prompt][f"accurate{method}"] = accurate

                    sc = False
                    for sent in split_sentences(desc.text):
                        for ext_sent in sentences_dict[sent]:
                            if ext_sent.tag == "strong":
                                sc = True
                    ds_by_prompt[desc.prompt][f"sc{method}"] = sc

        ds_by_popularity = sorted(
            (
                (int(d["s_pop"]), d)
                for d in ds_by_prompt.values()
                if d.get("accurate") is not None
                and d.get(f"accurate_{METHOD}") is not None
            ),
            key=lambda x: x[0],
        )
        total = len(ds_by_popularity)
        max_threshold = 0
        max_ratio = 0
        for i in range(len(ds_by_popularity)):
            accurate_method = sum(
                d.get(f"accurate_{METHOD}", 0) for _, d in ds_by_popularity[:i]
            )
            accurate_vanilla = sum(d["accurate"] for _, d in ds_by_popularity[i:])
            accurate_ratio = (accurate_method + accurate_vanilla) / total
            # print(accurate_ratio)
            if accurate_ratio > max_ratio:
                max_ratio = accurate_ratio
                max_threshold = i
        threshold_popularity = ds_by_popularity[max_threshold][0]
        # print(
        #     "Threshold popularity (%):",
        #     threshold_popularity,
        #     f"({max_threshold / total:.3f})",
        # )
        # print("Max accuracy:", max_ratio)


        for temp in [
            "popqa",
            "contriever",
        ]:
            annotations_file = f"annotations/popqa/{model}/reconciled/{temp}.json"
            with open(annotations_file) as f:
                annotations = json.load(f)

            for annotation in annotations:
                file = annotation["fn"]
                if file == "demo":
                    continue
                abs_path = Path(file).absolute()
                if abs_path not in fn_to_prompt:
                    continue
                prompt = fn_to_prompt[abs_path]
                ds_by_prompt[prompt][
                    "gt" + (f"_{temp}" if temp == "contriever" else "")
                ] = ("strong" in annotation["label"])

        print("Vanilla %SC: {:.1f}".format(100 * sum(d["sc"] for _, d in ds_by_popularity) / total))
        print(
            "Vanilla precision (%): {:.1f}".format(
            100 *
            sum(d.get("gt", False) for _, d in ds_by_popularity)
            / sum(d.get("sc", False) for _, d in ds_by_popularity),
        ))
        print(
            "Augmented %SC: {:.1f}".format(
                100 *
                sum(
                    d[f"sc{method if d['s_pop'] < threshold_popularity else ''}"]
                    for _, d in ds_by_popularity
                )
                / total,
                ))
        print(
            "Augmented precision (%): {:.1f}".format(
            100 *
            sum(
                d.get("gt" + (f"_{METHOD}" if p < threshold_popularity else ""), False)
                for p, d in ds_by_popularity
            )
            / sum(
                d.get("sc" + (f"_{METHOD}" if p < threshold_popularity else ""), False)
                for p, d in ds_by_popularity
            ),
        ))


if __name__ == "__main__":
    main()
