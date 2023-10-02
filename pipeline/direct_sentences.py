import json
import pathlib
import random

import chatprotect.consistency
import chatprotect.selfcheckgpt
from chatprotect import util
from chatprotect.util import read_sent_file, fetch_model, prompt_identifier
import logging

_LOGGER = logging.getLogger(__name__)

from argparse import ArgumentParser

RANDOM = 0
ASK = 1
ASK_COT = 3
ASK_COT_ORIGINAL = 33
ASK_COT_EXPLAIN_ONLY_MAJORITY = 4
ASK_SELFCHECK_BERTSCORE = 5
ASK_SELFCHECK_MQAG = 6
ASK_SELFCHECK_PPL = 7
ASK_NONFACTUAL = 8
ASK_COT_SCORE = 9
ASK_STEP_BY_STEP = 10
ASK_SELFCHECK_UNIAVG = 40
ASK_SELFCHECK_UNIMAX = 41

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
        default="test/custom/sentences/chatgpt",
    )
    ap.add_argument(
        "--mode",
        type=int,
        help="select a sentence comparison function to use",
        default=ASK_COT,
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

    if args.mode in (40, 41):
        sents = []
        for test_file in sorted(test_dir.iterdir()):
            if test_file.is_dir():
                continue
            if args.new not in test_file.stem:
                continue
            try:
                ext_sent = read_sent_file(target, test_file)
            except ValueError:
                continue
            if ext_sent.triple[1] != "compactie":
                continue
            s1, s2, prefix, tag = (
                ext_sent.orig,
                ext_sent.alt,
                ext_sent.prefix,
                ext_sent.tag,
            )
            sents.append(s1)
            sents.append(s2)
        chatprotect.selfcheckgpt.train(args, sents)

    for test_file in sorted(test_dir.iterdir()):
        if test_file.is_dir():
            continue
        if args.new not in test_file.stem:
            continue
        try:
            ext_sent = read_sent_file(target, test_file)
        except ValueError:
            continue
        if ext_sent.triple[1] != "compactie":
            continue
        s1, s2, prefix, tag = ext_sent.orig, ext_sent.alt, ext_sent.prefix, ext_sent.tag

        _LOGGER.debug(test_file)
        cost_before = sum(
            u.prompt_tokens + u.completion_tokens for u in bot.total_usage
        )
        try:
            if args.mode < 5 and s1 == s2:
                res = util.CHECKED
            else:
                if args.mode == RANDOM:
                    res = random.random()
                elif args.mode == ASK:
                    res = chatprotect.consistency.check_consistent_ask_direct(
                        bot, s1, s2, target, prefix
                    )
                elif args.mode == ASK_COT:
                    reason = chatprotect.consistency.explain_consistent_cot(
                        bot, s1, s2, target, prefix
                    )[0]
                    res = chatprotect.consistency.check_consistent_cot(
                        bot, s1, s2, target, prefix, reason
                    )
                elif args.mode == ASK_COT_ORIGINAL:
                    # This variant only works with prompts of the form "Tell me about ..."
                    reason = chatprotect.consistency.explain_consistent_cot_original(
                        bot, s1, s2, target, prefix
                    )[0]
                    res = chatprotect.consistency.check_consistent_cot_original(
                        bot, s1, s2, target, prefix, reason
                    )
                elif args.mode == ASK_COT_EXPLAIN_ONLY_MAJORITY:
                    res = chatprotect.consistency.check_consistent_cot_multivote(
                        bot, s1, s2, target, prefix
                    )
                elif args.mode == ASK_NONFACTUAL:
                    res = chatprotect.consistency.check_nonfactual_cot(
                        bot, s1, s2, target, prefix
                    )
                elif args.mode == ASK_COT_SCORE:
                    res = chatprotect.consistency.check_consistent_score(
                        bot, s1, s2, target, prefix
                    )
                elif args.mode == ASK_STEP_BY_STEP:
                    res = chatprotect.consistency.check_consistent_step_by_step(
                        bot, s1, s2, target, prefix
                    )
                else:
                    res = chatprotect.selfcheckgpt.check(args, s1, s2, target, prefix)
        except Exception as e:
            _LOGGER.error(
                "unrecoverable error trying to evaluate the statement, skipping",
                exc_info=e,
            )
            continue
        cost_after = sum(u.prompt_tokens + u.completion_tokens for u in bot.total_usage)
        print(
            json.dumps(
                {
                    "orig": s1,
                    "label": tag,
                    "score": res,
                    "cost": cost_after - cost_before,
                    "file": str(test_file),
                }
            ),
            flush=True,
        )
    print(
        json.dumps({k: v for k, v in zip(["TP", "FP", "TN", "FN"], inconsistency_mat)})
    )
