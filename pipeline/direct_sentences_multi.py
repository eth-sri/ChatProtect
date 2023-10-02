import json
import pathlib
import random
from collections import defaultdict

import chatprotect.consistency
import chatprotect.selfcheckgpt
from chatprotect.util import read_sent_file, fetch_model, prompt_identifier, ExtSentence
import logging

_LOGGER = logging.getLogger(__name__)

from argparse import ArgumentParser

ASK_COT_FACTUAL_MULTI = 20
ASK_SELFCHECK_BERTSCORE_MULTI = 12
ASK_SELFCHECK_MQAG_MULTI = 13
ASK_SELFCHECK_PPL_MULTI = 14
ASK_SELFCHECK_UNIAVG_MULTI = 42
ASK_SELFCHECK_UNIMAX_MULTI = 43

CONTINUATIONS_OF_PREFIX = 0
CONTINUATIONS_OF_PREFIX_W_REPHRASE = 1
ANSWER_QUESTION = 2
CONTINUATIONS_OF_PREFIX_W_TRIPLE = 3

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--prompt",
        type=str,
        help="Target of the inconsistency check",
        default="Please tell me about Thomas Chapais",
    )
    ap.add_argument(
        "--test_sentence_dir",
        type=str,
        help="directory containing test sentences",
        default="test/test/sentences/chatgpt",
    )
    ap.add_argument(
        "--mode",
        type=int,
        help="select a sentence comparison function to use",
        default=ASK_COT_FACTUAL_MULTI,
    )
    ap.add_argument(
        "--sent-mode",
        type=int,
        help="select a sentence generation function to use",
        default=CONTINUATIONS_OF_PREFIX_W_TRIPLE,
    )
    ap.add_argument(
        "--model",
        type=str,
        help="select a model to run the test against",
        default="chatgpt",
    )
    ap.add_argument(
        "--lm",
        type=str,
        help="lm size for computing ppl",
        default="1.3b",
    )
    ap.add_argument(
        "--new",
        type=str,
        help="test only new method",
        default="",
    )
    ap.add_argument(
        "--alts",
        type=int,
        help="number of alternative sentences to use in one run",
        default=2,
    )
    args = ap.parse_args()
    model = args.model
    bot = fetch_model(model)
    random.seed(0)
    sent_mode = args.sent_mode

    target: str = args.prompt
    test_dir = (
        pathlib.Path(args.test_sentence_dir)
        / prompt_identifier(target)
        / f"m{sent_mode}"
    )
    print(
        f"Processing {len(list(test_dir.iterdir()))} files, writing results to stdout"
    )
    inconsistencies, spurious, ok = 0, 0, 0
    TP, FP, TN, FN = 0, 1, 2, 3
    inconsistency_mat = [
        0,
        0,
        0,
        0,
    ]  # TP (strong), FP (ok), TN (ok), FN (strong)

    ext_sents = defaultdict(list)
    for test_file in sorted(test_dir.iterdir()):
        if test_file.is_dir():
            continue
        if args.new not in test_file.stem:
            continue
        try:
            ext_sent = read_sent_file(target, test_file)
        except ValueError:
            continue
        ext_sents[ext_sent.orig].append(ext_sent)

    if args.mode in (42, 43):
        sents = []
        for ext_sent_list in ext_sents.values():
            sents.append(ext_sent_list[0].orig)
            sents += [e.alt for e in ext_sent_list][: args.alts]
        chatprotect.selfcheckgpt.train(args, sents)

    for ext_sent_list in ext_sents.values():
        ext_sent_by_triple = defaultdict(list)
        for e in ext_sent_list:
            ext_sent_by_triple[tuple(e.triple[0])].append(e)
        cost_before = sum(
            u.prompt_tokens + u.completion_tokens for u in bot.total_usage
        )
        try:
            # filter out same number of triples
            if args.mode == ASK_COT_FACTUAL_MULTI:
                stmt1e: ExtSentence = ext_sent_list[0]
                stmt2s = [e.alt for e in ext_sent_list][: args.alts]
                res = chatprotect.consistency.check_factual_multi_score(
                    bot, stmt1e.orig, stmt2s, target, stmt1e.prefix
                )
            else:
                stmt1e: ExtSentence = ext_sent_list[0]
                stmt2s = [e.alt for e in ext_sent_list][: args.alts]
                res = chatprotect.selfcheckgpt.check_multi(
                    args, stmt1e.orig, stmt2s, target, stmt1e.prefix
                )
        except Exception as e:
            _LOGGER.error(
                "unrecoverable error trying to evaluate the statement, skipping",
                exc_info=e,
            )
            continue
        cost_after = sum(u.prompt_tokens + u.completion_tokens for u in bot.total_usage)
        ext_sent_sample = ext_sent_list[0]
        print(
            json.dumps(
                {
                    "orig": ext_sent_sample.orig,
                    "wrong": ext_sent_sample.wrong,
                    "score": res,
                    "cost": cost_after - cost_before,
                }
            )
        )
    print(
        json.dumps({k: v for k, v in zip(["TP", "FP", "TN", "FN"], inconsistency_mat)})
    )
