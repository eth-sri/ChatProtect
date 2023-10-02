import json
import pathlib
from argparse import ArgumentParser
from collections import defaultdict
import editdistance

from chatprotect.util import read_sent_file, read_desc_file, split_sentences

testset = "test"
mode = 3
alm_model = "chatgpt"
glm_model = "chatgpt"
print("Original", 0)
until_map = {
    1: "Naively drop",
    2: "Iteration 1",
    3: "Iteration 2",
    4: "Iteration 3",
    5: "Final (ours)",
}
for until in (
    2,
    3,
    4,
    5,
    1,
):

    test_sentence_dir = pathlib.Path(f"test/{testset}/sentences") / glm_model
    test_description_dir = pathlib.Path(f"test/{testset}/descriptions") / glm_model
    test_new_description_dir = (
        pathlib.Path(f"test/{testset}/new_descriptions") / alm_model / glm_model
    )
    ext_sent_dict = defaultdict(list)
    for target_dir in sorted(test_sentence_dir.iterdir()):
        test_sentence_dir = target_dir / f"m{mode}"
        target_id = target_dir.name
        for test_file in sorted(test_sentence_dir.iterdir()):
            try:
                ext_sent = read_sent_file(None, test_file)
            except ValueError:
                continue
            if ext_sent.triple[1] != "compactie":
                continue
            ext_sent.file = test_file
            ext_sent_dict[ext_sent.orig].append(ext_sent)
    ext_sent_output_dict = {}
    has_contr_total = 0
    for target_dir, target_dir_type in sorted(
        (x, "orig") for x in test_description_dir.iterdir()
    ):
        for test_file in sorted(target_dir.iterdir()):
            try:
                desc = read_desc_file(None, test_file)
            except ValueError:
                continue
            split_desc = split_sentences(desc.text)
            for i, sent in enumerate(split_desc):
                triples = ext_sent_dict[sent]
                has_contr = 0
                for ext_sent in triples:
                    if ext_sent.tag != "ok":
                        has_contr += 1
                if has_contr > 0:
                    has_contr_total += 1
    has_contr_final = 0
    for target_dir, target_dir_type in [
        (subdir / f"m{until}", "new") for subdir in test_new_description_dir.iterdir()
    ]:
        for test_file in sorted(target_dir.iterdir()):
            try:
                desc = read_desc_file(None, test_file)
            except ValueError:
                continue
            split_desc = split_sentences(desc.text)
            for i, sent in enumerate(split_desc):
                triples = ext_sent_dict[sent]
                has_contr = 0
                for ext_sent in triples:
                    if ext_sent.tag != "ok":
                        has_contr += 1
                if has_contr > 0:
                    has_contr_final += 1
    print(until_map[until], f"{(1-has_contr_final/has_contr_total)*100:.1f}")
