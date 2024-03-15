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

for testset in ["test0.00", "test0.25", "test0.50", "test0.75"]:

    mode = 3
    alm_model = "chatgpt"
    glm_model = "chatgpt"
    ENTITIES = read_entities(f"./test/{testset}/entities.txt")
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
    # print(f"{testset[4:]}, {sum(ok)}, {sum(contr)}, {sum(contr)/sum(ok):.3f}")
    print(f"{testset[4:]}, {100 * sum(contr)/sum(ok):.1f}")

# for the old results we need a seperate script
import os
import argparse
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.metrics import (
    PrecisionRecallDisplay,
)

from chatprotect.util import read_sent_file, read_entities, prompt_identifier


def get_args():
    args = argparse.Namespace()
    args.test_dir = "output/test/chatgpt/chatgpt/s3/test_m3/"
    args.old = True
    args.total = True
    args.override_sent = True
    args.mode = "contradiction"
    args.method = "compactie"
    args.test_sentence_dir = "test/test/sentences/chatgpt/"
    args.sent_mode = 3
    return args


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
    for ent_path in sorted(os.listdir(test_dir)):
        if any(ent_path.startswith(s) for s in ("m4", "m3", "m2")):
            continue
        local_scores, local_labels, local_files, local_convos = [], [], [], []
        local_convo = ""
        with open(os.path.join(test_dir, ent_path)) as f:
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
    sents_with_sc = {}
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
            sents_with_sc[ext_sent.orig] = sents_with_sc.get(ext_sent.orig, 0) + (
                score > 0.5
            )
            if args.override_sent:
                if args.mode == "contradiction":
                    label = int(ext_sent.tag in ("strong", "weak"))
                else:
                    label = int(ext_sent.wrong in ("both", "1"))

        derived_label = int(score > 0.5)
        if derived_label == 1 and label == 1:
            inconsistency_mat[TP] += 1
        if derived_label == 1 and label == 0:
            inconsistency_mat[FP] += 1
        if derived_label == 0 and label == 0:
            inconsistency_mat[TN] += 1
        if derived_label == 0 and label == 1:
            inconsistency_mat[FN] += 1

    if args.total:
        sents_with_sc_sum = sum(1 for x in sents_with_sc.values() if x > 0)
        # print(
        #     f"1.00, {len(sents_with_sc)}, {sents_with_sc_sum}, {sents_with_sc_sum / len(sents_with_sc):.3f}"
        # )
        print(
            f"1.00, {100 * sents_with_sc_sum / len(sents_with_sc):.1f}"
        )


if __name__ == "__main__":
    main()
