"""
Full pipeline for demonstration purposes
"""
import argparse
from collections import defaultdict
from typing import Optional

from nltk import sent_tokenize

from chatprotect.consistency import check_consistent_cot, explain_consistent_cot
from chatprotect.rephrase import (
    rephrase_desc_remove_conflict,
    rephrase_desc_without_sentence,
)
from chatprotect.sentences import (
    generate_statement_missing_object,
    extract_triples_compact_ie,
)
from chatprotect.util import ExtSentence, label_to_tag, fetch_model
from chatprotect.bots import bot


def main(target: str, alm_bot: bot.Bot, glm_bot: bot.Bot, steps: Optional[int] = 3):
    # Step 1: Generate a description for target from the LLM
    with glm_bot as bot_ses:
        bot_ses.set_deterministic(False)
        bot_ses.set_num_answers(1)
        desc = bot_ses.ask(target)[0]

    print("******* Initial Description *******")
    print(desc)
    # repeat until convergence or fixed number of steps
    if steps is None:
        steps = 1000000
    for i in range(steps + 1):
        print("******* Checking for hallucinations *******")
        # Step 2: Check for each sentence how sure the LLM is about it (generate alt sentence and check for consistency)
        sentences = sent_tokenize(desc)
        sent_tagged_dict = defaultdict(list)
        prefix = ""
        ok, incons = 0, 0
        for sentence in sentences:
            # if same sentence as in original description, pass
            if sentence in sent_tagged_dict:
                for ext_sent in sent_tagged_dict[sentence]:
                    if ext_sent.tag == "ok":
                        ok += 1
                    else:
                        incons += 1
                continue
            triples = extract_triples_compact_ie(sentence)

            for triple in triples:
                alt_statement = generate_statement_missing_object(
                    glm_bot, triple[0], triple[1], target, prefix
                )[0]
                explanation = explain_consistent_cot(
                    alm_bot, sentence, alt_statement, target, prefix
                )[0]
                label = check_consistent_cot(
                    alm_bot, sentence, alt_statement, target, prefix, explanation
                )
                tag = label_to_tag(label)
                sent_tagged_dict[sentence].append(
                    ExtSentence(
                        target,
                        sentence,
                        alt_statement,
                        prefix,
                        triple,
                        tag,
                        explanation=explanation,
                    )
                )
                if tag == "ok":
                    ok += 1
                else:
                    incons += 1
            prefix = f"{prefix} {sentence}"
        print(
            f"Description contains {ok} checked facts and {incons} detected hallucinations."
        )
        if incons == 0:
            break

        # Step 3: Generate a new description using the extracted sentences and triples
        print("******* Revising description *******")
        if i == steps:
            print(f"Dropping sentence with {incons} detected hallucinations.")
            desc, decisions = rephrase_desc_without_sentence(
                alm_bot, desc, sent_tagged_dict, target
            )
        else:
            print(f"Removing {incons} detected hallucinations from sentences.")
            desc, decisions = rephrase_desc_remove_conflict(
                alm_bot, desc, sent_tagged_dict, target
            )
    print("******* Final description *******")
    print(desc)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--prompt",
        default="Please tell me about Thomas Chapais",
        type=str,
        help="Target of the hallucination check",
    )
    a.add_argument(
        "--alm",
        default="chatgpt",
        type=str,
        help="Used LLM as aLM (i.e. ChatGPT, GPT4 or Vicuna-13B)",
    )
    a.add_argument(
        "--glm",
        default="chatgpt",
        type=str,
        help="Used LLM as gLM (i.e. ChatGPT, GPT4 or Vicuna-13B)",
    )
    a.add_argument(
        "--iterations",
        default=3,
        type=int,
        help="Number of iterations before the iterative removal stops and simply drops the remaining hallucinations.",
    )
    args = a.parse_args()
    alm_bot = fetch_model(args.alm)
    glm_bot = fetch_model(args.glm)
    main(args.prompt, alm_bot, glm_bot, args.iterations)
