import json
from argparse import ArgumentParser
import concurrent.futures
import pathlib
from collections import defaultdict
from dataclasses import asdict

from chatprotect.util import (
    read_desc_file,
    split_sentences,
    read_sent_file,
    ExtSentence,
    SentCost,
    fetch_model,
    prompt_identifier,
)
from chatprotect.consistency import check_consistent_cot, explain_consistent_cot
from chatprotect.sentences import (
    generate_alt_statements_prefix_with_triple_wo_object,
    generate_alt_statements_answer_question,
    generate_alt_statement_prefix,
    generate_alt_statement_prefix_rephrase,
    generate_statement_missing_object_free,
)

import logging

_LOGGER = logging.getLogger(__name__)

CONTINUATIONS_OF_PREFIX = 0
CONTINUATIONS_OF_PREFIX_W_REPHRASE = 1
ANSWER_QUESTION = 2
CONTINUATIONS_OF_PREFIX_W_TRIPLE = 3
CONTINUATIONS_OF_PREFIX_W_TRIPLE_FREE = 4

parser = ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    help="Prompt to generate alternative sentences for",
    default="Please tell me about Thomas Chapais",
)
parser.add_argument(
    "--test_description_dir",
    type=str,
    help="Input dir",
    default="./test/custom/descriptions",
)
parser.add_argument(
    "--test_sentence_dir",
    type=str,
    help="Output dir",
    default="./test/custom/sentences",
)
parser.add_argument(
    "--glm-model",
    type=str,
    help="select a text generation model",
    default="chatgpt",
)
parser.add_argument(
    "--alm-model",
    type=str,
    help="select a text analysis model",
    default="chatgpt",
)
parser.add_argument(
    "--mode",
    type=int,
    help="which mode to operate in",
    default=CONTINUATIONS_OF_PREFIX_W_TRIPLE,
)
parser.add_argument(
    "--workers",
    type=int,
    help="Fetch sentences in parallel for this amount of workers",
    default=1,
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite existing sentence pairs",
)
parser.add_argument(
    "--no-check",
    action="store_true",
    help="do not check the contradictoriness",
)
parser.add_argument(
    "--new",
    type=str,
    default="",
    help="The descriptions are the result of a redaction step",
)
parser.add_argument(
    "--alts",
    type=int,
    default=1,
    help="The number of alternatives sentences sampled per sentence",
)
parser.add_argument(
    "--override-temperature",
    type=int,
    default=None,
    help="The temperature used for sampling, set to override default deterministic behavior",
)
args = parser.parse_args()

model = args.alm_model
sent_model = args.glm_model

target = args.prompt
workers = args.workers
mode = args.mode
target_id = prompt_identifier(target)
output_dir = pathlib.Path(args.test_sentence_dir) / sent_model / target_id / f"m{mode}"
if output_dir.is_dir() and not args.force and not args.new:
    print("Directory exists already, skipping")
    exit()
output_dir.mkdir(exist_ok=True, parents=True)
new_desc = args.new
if new_desc:
    input_dir = pathlib.Path(args.test_description_dir) / model / sent_model / target_id
else:
    input_dir = pathlib.Path(args.test_description_dir) / sent_model / target_id

# if we are treating descriptions that have been redacted, the existing files can be re-used
if new_desc:
    # parse sentences
    sents = defaultdict(list)
    for test_file in sorted(output_dir.iterdir()):
        try:
            ext_sent = read_sent_file(target, test_file)
            sents[ext_sent.orig] = ext_sent
        except ValueError:
            continue
else:
    sents = {}


def process_description_file(input_file: pathlib.Path):
    try:
        description = read_desc_file(target, input_file).text
    except ValueError:
        return
    tag_bot = fetch_model(model)
    bot = fetch_model(sent_model)

    if new_desc:
        additional_file_id = f"{input_file.parent.stem}_"
        if sent_model != model:
            additional_file_id += f"{model}_"
    else:
        additional_file_id = ""

    sentences = split_sentences(description)
    prefix = ""
    for i, sentence in enumerate(sentences):
        if sentence in sents:
            # skip those sentences that we already covered by original description
            alt_statements = []
        elif mode == CONTINUATIONS_OF_PREFIX_W_TRIPLE:
            alt_statements = generate_alt_statements_prefix_with_triple_wo_object(
                bot, target, sentence, prefix, args.alts, args.override_temperature
            )
        elif mode == CONTINUATIONS_OF_PREFIX_W_TRIPLE_FREE:
            alt_statements = generate_alt_statements_prefix_with_triple_wo_object(
                bot,
                target,
                sentence,
                prefix,
                args.alts,
                args.override_temperature,
                generate_statement_missing_object_free,
            )
        elif mode == CONTINUATIONS_OF_PREFIX:
            alt_statements = generate_alt_statement_prefix(bot, target, prefix, 2)
        elif mode == CONTINUATIONS_OF_PREFIX_W_REPHRASE:
            alt_statements = generate_alt_statement_prefix_rephrase(
                bot, target, sentence, prefix, 1
            )
        elif mode == ANSWER_QUESTION:
            alt_statements = generate_alt_statements_answer_question(
                bot, target, sentence, prefix
            )
        else:
            raise NotImplementedError("unknown mode")

        for j, (alt_statement, aux) in enumerate(alt_statements):
            local_highest_number = f"{additional_file_id}{input_file.stem}_{i}_{j}"
            if args.no_check:
                exp = ""
                preliminary_label = "ok"
            else:
                exp = explain_consistent_cot(
                    tag_bot, sentence, alt_statement, target, prefix
                )[0]
                preliminary_tag = check_consistent_cot(
                    tag_bot, sentence, alt_statement, target, prefix, exp
                )
                if preliminary_tag > 0.5:
                    preliminary_label = "strong"
                    _LOGGER.info(f"{local_highest_number}: {preliminary_label}")
                else:
                    preliminary_label = "ok"
                    _LOGGER.info(f"{local_highest_number}: {preliminary_label}")
            e = ExtSentence(
                target=target,
                orig=sentence,
                alt=alt_statement,
                prefix=prefix,
                triple=aux,
                tag=preliminary_label,
                wrong="unk",
                explanation=exp,
            )
            with (output_dir / (str(local_highest_number) + ".txt")).open("w") as f:
                json.dump(asdict(e), f, indent=2)
        prefix = f"{prefix} {sentence}"
    sc = SentCost(
        target,
        cost_user=sum(u.prompt_tokens for u in bot.total_usage),
        cost_model=sum(u.completion_tokens for u in bot.total_usage),
    )
    with (output_dir / f"_cost_{additional_file_id}{input_file.stem}.txt").open(
        "w"
    ) as f:
        json.dump(asdict(sc), f, indent=2)


input_files = sorted(input_dir.iterdir())
if new_desc:
    input_files = sum(
        (sorted(d.iterdir()) for d in input_files if args.new == d.stem), []
    )
print(f"Processing {len(input_files)} files and writing sentence pairs to {output_dir}")
if workers == 1:
    for input_file in input_files:
        process_description_file(input_file)
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_description_file, input_file)
            for input_file in input_files
        ]
