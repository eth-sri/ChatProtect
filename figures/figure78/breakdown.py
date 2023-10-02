import json
import pathlib
from argparse import ArgumentParser
from collections import defaultdict
import editdistance

from chatprotect.util import (
    read_sent_file,
    read_desc_file,
    split_sentences,
    read_entities,
    prompt_identifier,
)

testset = "test_big"
mode = 3
alm_model = "chatgpt"
glm_model = "chatgpt"
ENTITIES = read_entities(f"./test/{testset}/entities_by_popularity.txt")
ext_sent_dict = defaultdict(list)
test_sentence_dir = pathlib.Path(f"test/{testset}/sentences") / glm_model
test_description_dir = pathlib.Path(f"test/{testset}/descriptions") / glm_model
for target_dir in sorted(test_sentence_dir.iterdir()):
    test_sentence_subdir = target_dir / f"m{mode}"
    target_id = target_dir.name
    for test_file in sorted(test_sentence_subdir.iterdir()):
        try:
            ext_sent = read_sent_file(None, test_file)
        except ValueError:
            continue
        if ext_sent.triple[1] != "compactie":
            continue
        ext_sent.file = test_file
        ext_sent_dict[ext_sent.orig].append(ext_sent)
ok = []
contr = []
for target in ENTITIES:
    target = f"Please tell me about {target}"
    target_id = prompt_identifier(target)

    target_dir = test_description_dir / target_id
    contr_sum = 0
    total_sum = 0
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
                contr_sum += 1
            total_sum += 1
    ok.append(total_sum)
    contr.append(contr_sum)
for range in ((0, 50), (50, 100)):
    for color, list, ex in (
        ("mygreen", ok, ""),
        (
            "myred",
            contr,
            r", nodes near coords, every node near coord/.append style={font=\tiny}",
        ),
    ):
        print(rf"\addplot[fill={color}{ex}] coordinates {{")
        for i, e in enumerate(list[range[0] : range[1]], start=range[0]):
            print(f"({e},{100-i})")
        print(r"};")
