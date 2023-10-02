import json
import pathlib
from argparse import ArgumentParser
from collections import defaultdict
import editdistance

from chatprotect.util import read_sent_file, read_desc_file, split_sentences

testset = "test"
glm_model = "chatgpt"
alm_model = "chatgpt"
mode_map = {
    0: "Continue",
    1: "Rephrase",
    2: "Q&A",
    3: "Ours",
}
for mode in (0, 1, 2, 3):

    test_sentence_dir = pathlib.Path(f"test/{testset}/sentences") / glm_model
    test_description_dir = pathlib.Path(f"test/{testset}/descriptions") / glm_model
    ext_sent_dict = defaultdict(list)
    for target_dir in sorted(test_sentence_dir.iterdir()):
        test_sentence_dir = target_dir / f"m{mode}"
        target_id = target_dir.name
        for test_file in sorted(test_sentence_dir.iterdir()):
            try:
                ext_sent = read_sent_file(None, test_file)
            except ValueError:
                continue
            if mode == 3 and ext_sent.triple[1] != "compactie":
                continue
            ext_sent.file = test_file
            ext_sent_dict[ext_sent.orig].append(ext_sent)
    ext_sent_output_dict = {}
    has_contr_total = 0
    total = 0
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
                total += 1
                triples = ext_sent_dict[sent]
                has_contr = 0
                for ext_sent in triples:
                    if ext_sent.tag != "ok":
                        has_contr += 1
                if has_contr > 0:
                    has_contr_total += 1
    print(mode_map[mode], f"{has_contr_total*100/total:.1f}")
