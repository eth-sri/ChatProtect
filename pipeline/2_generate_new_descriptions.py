import json
import pathlib
from collections import defaultdict
import concurrent.futures
from dataclasses import asdict

import chatprotect.rephrase
from chatprotect.util import (
    read_sent_file,
    read_desc_file,
    Description,
    Decisions,
    fetch_model,
    prompt_identifier,
)

import logging

_LOGGER = logging.getLogger(__name__)

from argparse import ArgumentParser

ORGINAL_DESC = 0
OMIT_SENTENCE = 1
OMIT_TRIPLE_TRANSITION_IMPROVED_1_STEP = 2
OMIT_TRIPLE_TRANSITION_IMPROVED_2_STEP = 3
OMIT_TRIPLE_TRANSITION_IMPROVED_3_STEP = 4
OMIT_REMAINING_SENTENCE = 5

ap = ArgumentParser()
ap.add_argument(
    "--prompt",
    type=str,
    help="Prompt to generate new descriptions for",
    default="Please tell me about Thomas Chapais",
)
ap.add_argument(
    "--test_set",
    type=str,
    help="test set name",
    default="custom",
)
ap.add_argument(
    "--test_sentence_dir",
    type=str,
    help="directory containing test sentences",
    default="test/custom/sentences",
)
ap.add_argument(
    "--test_description_dir",
    type=str,
    help="directory containing test descriptions",
    default="test/custom/descriptions",
)
ap.add_argument(
    "--test_new_description_dir",
    type=str,
    help="target directory containing newly generated descriptions",
    default="test/custom/new_descriptions",
)
ap.add_argument(
    "--test_decision_record_dir",
    type=str,
    help="target directory containing decision_records",
    default="test/custom/decisions",
)
ap.add_argument(
    "--mode",
    type=int,
    help="select a description generation function to use",
    default=OMIT_TRIPLE_TRANSITION_IMPROVED_1_STEP,
)
ap.add_argument(
    "--glm-model",
    type=str,
    help="select a text generation model",
    default="chatgpt",
)
ap.add_argument(
    "--alm-model",
    type=str,
    help="select a text analysis model",
    default="chatgpt",
)
ap.add_argument(
    "--method",
    type=str,
    help="select a method for triple extraction",
    choices=["both", "native", "compactie"],
    default="compactie",
)
ap.add_argument(
    "--workers",
    type=int,
    help="Fetch sentences in parallel for this amount of workers",
    default=1,
)
ap.add_argument(
    "--sentence-generation-method",
    type=int,
    help="Method by which alternative sentences were generated",
    default=3,
)
ap.add_argument(
    "--old",
    action="store_true",
)
args = ap.parse_args()

model = args.alm_model
bot = fetch_model(model)
sentence_model = args.glm_model

target: str = args.prompt
testset = args.test_set
workers = args.workers
target_id = prompt_identifier(target)
method = args.method
test_sentence_dir = (
    pathlib.Path(args.test_sentence_dir)
    / sentence_model
    / target_id
    / f"m{args.sentence_generation_method}"
)
test_output_files = [
    pathlib.Path(
        f"output/{testset}/{model}/{sentence_model}/s3/test_m{args.sentence_generation_method}"
    )
    / f"{mx}{target_id}.chatprotect"
    for mx in ["m4", "m3", "m2", ""]
]
if OMIT_TRIPLE_TRANSITION_IMPROVED_2_STEP <= args.mode:
    test_description_dir = (
        pathlib.Path(args.test_description_dir) / model / sentence_model / target_id
    )
else:
    test_description_dir = (
        pathlib.Path(args.test_description_dir) / sentence_model / target_id
    )
test_new_description_dir = (
    pathlib.Path(args.test_new_description_dir) / model / sentence_model / target_id
)
test_decision_record_dir = (
    pathlib.Path(args.test_decision_record_dir) / model / sentence_model / target_id
)
inconsistencies, spurious, ok = 0, 0, 0
TP, FP, TN, FN = 0, 1, 2, 3
inconsistency_mat = [
    0,
    0,
    0,
    0,
]  # TP (strong), FP (ok), TN (ok), FN (strong)

# parse sentences
sents_by_file = {}
for test_file in sorted(test_sentence_dir.iterdir()):
    try:
        ext_sent = read_sent_file(target, test_file)
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
        if args.old:
            if line.startswith("score:"):
                scores.append(float(line[line.find(" ") + 1 :]))
            elif line.startswith(str(test_sentence_dir)):
                files.append(pathlib.Path(line))
        else:
            try:
                d = json.loads(line)
            except:
                continue
            if "score" not in d:
                continue
            files.append(pathlib.Path(d["file"]))
            scores.append(d["score"])
    assert len(scores) == len(files)
    # overwrite tag with detected score
    for score, file in zip(scores, files):
        if str(file.absolute()) not in sents_by_file:
            continue
        sents_by_file[str(file.absolute())].tag = "ok" if score <= 0.5 else "strong"
# restore original structure of sent dict
sents = defaultdict(list)
for sent in sents_by_file.values():
    if method != "both" and method != sent.triple[1]:
        continue
    sents[sent.orig].append(sent)


def process_description_file(test_file: pathlib.Path):
    try:
        desc = read_desc_file(target, test_file).text
    except ValueError:
        return

    _LOGGER.debug(test_file)
    try:
        decision_record = []
        if args.mode == ORGINAL_DESC:
            res = desc
        elif args.mode in (OMIT_SENTENCE, OMIT_REMAINING_SENTENCE):
            res = chatprotect.rephrase.rephrase_desc_without_sentence(
                bot,
                desc,
                sents,
                target,
            )
        elif args.mode in (
            OMIT_TRIPLE_TRANSITION_IMPROVED_1_STEP,
            OMIT_TRIPLE_TRANSITION_IMPROVED_2_STEP,
            OMIT_TRIPLE_TRANSITION_IMPROVED_3_STEP,
        ):
            (
                res,
                decision_record,
            ) = chatprotect.rephrase.rephrase_desc_remove_conflict(
                bot,
                desc,
                sents,
                target,
            )
        else:
            print("unknown mode")
            exit(-1)
    except Exception as e:
        _LOGGER.error(
            "unrecoverable error trying to evaluate the statement, skipping", exc_info=e
        )
        return
    (test_new_description_dir / ("m" + str(args.mode))).mkdir(
        exist_ok=True, parents=True
    )
    d = Description(
        prompt=target,
        cost_model=sum(u.prompt_tokens for u in bot.total_usage),
        cost_user=sum(u.completion_tokens for u in bot.total_usage),
        text=res,
    )
    with (test_new_description_dir / ("m" + str(args.mode)) / test_file.name).open(
        "w"
    ) as f:
        json.dump(asdict(d), f, indent=2)
    (test_decision_record_dir / ("m" + str(args.mode))).mkdir(
        exist_ok=True, parents=True
    )
    decisions = Decisions(target, decision_record)
    with (test_decision_record_dir / ("m" + str(args.mode)) / test_file.name).open(
        "w"
    ) as f:
        json.dump(asdict(decisions), f, indent=2)


input_files = sorted(test_description_dir.iterdir())
if (
    args.mode >= OMIT_TRIPLE_TRANSITION_IMPROVED_2_STEP
    and args.mode <= OMIT_REMAINING_SENTENCE
):
    input_files = sum(
        (sorted(d.iterdir()) for d in input_files if f"m{args.mode-1}" == d.stem), []
    )
print(
    f"Processing {len(input_files)} files and writing revised descriptions to {test_new_description_dir} (logging revision descriptions at {test_decision_record_dir})"
)
if workers == 1:
    for input_file in input_files:
        process_description_file(input_file)
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_description_file, input_file)
            for input_file in input_files
        ]
