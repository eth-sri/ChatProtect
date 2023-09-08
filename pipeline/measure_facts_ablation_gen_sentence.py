from argparse import ArgumentParser
import pathlib


from chatprotect.util import (
    read_desc_file,
    read_sent_file,
    split_sentences,
    read_entities,
    prompt_identifier,
)
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument("--set", type=str, help="testset to be assessed", default="test")
parser.add_argument(
    "--test_description_dir",
    type=str,
    help="Input dir",
    default="./test/custom/descriptions",
)
parser.add_argument(
    "--test_sentence_dir", type=str, help="Input dir", default="./test/custom/sentences"
)
parser.add_argument(
    "--mode",
    type=int,
    help="select a description generation function that was used. Negative for original description",
    default=-1,
)
parser.add_argument(
    "--sent-mode",
    type=int,
    help="select a sentence pair generation function that was used.",
    default=3,
)
parser.add_argument(
    "--alm-model",
    type=str,
    help="select a model to run the test against",
    default="chatgpt",
)
parser.add_argument(
    "--glm-model",
    type=str,
    help="select a model to run the test against",
    default="chatgpt",
)
parser.add_argument(
    "--method",
    type=str,
    help="select a method for triple extraction",
    choices=["both", "native", "compactie"],
    default="compactie",
)
parser.add_argument(
    "--ground-truth",
    action="store_true",
)
args = parser.parse_args()

test_set = args.set

model = args.alm_model
sent_model = args.glm_model

mode = args.mode
sent_mode = args.sent_mode

ENTITIES = read_entities(f"./test/{test_set}/entities.txt")
entities_rev_map = {
    e.lower().replace(" ", "_"): prompt_identifier(f"Please tell me about {e}")
    for e in ENTITIES
}
fact_total = [0, 0]
for target in ENTITIES:
    target_id = prompt_identifier(f"Please tell me about {target}")
    test_sentence_dir = (
        pathlib.Path(args.test_sentence_dir) / sent_model / target_id / f"m{sent_mode}"
    )
    test_output_files = (
        [
            pathlib.Path(f"output/{test_set}/{model}/{sent_model}/s2/test_m{sent_mode}")
            / f"{mx}{target.lower().replace(' ', '_')}.chatprotect"
            for mx in ["m4_", "m3_", "m2_", ""]
        ]
        if not args.ground_truth
        else []
    )
    method = args.method

    # parse sentences
    sents_by_file = {}
    for test_file in sorted(test_sentence_dir.iterdir()):
        if any(test_file.stem.startswith(s) for s in ("m4", "m3", "m2")):
            continue
        try:
            ext_sent = read_sent_file(f"Please tell me about {target}", test_file)
            sents_by_file[str(test_file.absolute())] = ext_sent
        except ValueError:
            continue
    # parse output file
    for test_output_file in test_output_files:
        if not test_output_file.is_file():
            continue
        with open(test_output_file) as f:
            lines = f.readlines()
        scores, files = [], []
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("score:"):
                scores.append(float(line[line.find(" ") + 1 :]))
            elif line.startswith(str(args.test_sentence_dir)):
                files.append(pathlib.Path(line))
        assert len(scores) == len(files)
        # overwrite tag with detected score
        for score, file in zip(scores, files):
            file_str = str(file.absolute())
            for rev_e, e in entities_rev_map.items():
                file_str = file_str.replace(rev_e, e)
            if file_str in sents_by_file:
                sents_by_file[file_str].tag = "ok" if score <= 0.5 else "strong"
    # restore original structure of sent dict
    sents = defaultdict(list)
    for sent in sents_by_file.values():
        if sent_mode == 3 and method != "both" and method != sent.triple[1]:
            continue
        sents[sent.orig].append(sent)

    input_texts = []
    OK, STRONG = 0, 1
    facts = [0, 0]
    if mode >= 0:
        desc_dir = (
            pathlib.Path(args.test_description_dir)
            / model
            / sent_model
            / target_id
            / f"m{mode}"
        )
    else:
        desc_dir = pathlib.Path(args.test_description_dir) / sent_model / target_id
    unique_sents = set()
    for desc_file in sorted(desc_dir.iterdir()):
        try:
            desc = read_desc_file(f"Please tell me about {target}", desc_file).text
        except ValueError:
            continue
        desc_sents = split_sentences(desc)

        for sent in desc_sents:
            unique_sents.add(sent)
    for sent in unique_sents:
        ext_sents = sents[sent]
        for ext_sent in ext_sents:
            tag = OK if ext_sent.tag == "ok" else STRONG
            facts[tag] += 1

    print(",".join(map(str, facts)))
    fact_total[0] += facts[0]
    fact_total[1] += facts[1]
print(",".join(map(str, fact_total)))
