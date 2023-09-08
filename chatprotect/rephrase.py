import typing


from chatprotect.util import ExtSentence, split_sentences


def rephrase_desc_without_sentence(
    bot,
    desc,
    incons_sents: typing.Dict[str, typing.List[ExtSentence]],
    target,
):
    """Drops all sentences from the description that contain at least one contradiction"""
    desc_sents = split_sentences(desc)
    desc_new = " ".join(
        s for s in desc_sents if all(e.tag == "ok" for e in incons_sents[s])
    )
    return desc_new, []


def rephrase_desc_remove_conflict(
    bot,
    desc,
    incons_sents: typing.Dict[str, typing.List[ExtSentence]],
    target,
):
    """Asks the model to remove information from a sentence that was deemed self-contradictory"""
    desc_sents = split_sentences(desc)
    decision_record = []
    new_desc = []
    prev_desc = []
    for sent in desc_sents:
        alt_sent = sent
        ext_version = incons_sents[sent]
        full_info = {
            "prefix_original": " ".join(prev_desc),
            "prefix_new": " ".join(new_desc),
            "sent_orig": sent,
        }
        prev_desc.append(sent)

        # skip sentence directly if there are several triples extracted and all triples are inconsistent
        if len(ext_version) > 1 and all(
            ext.tag in ("weak", "strong") for ext in ext_version
        ):
            full_info["decision"] = "drop"
            full_info["sent_new"] = ""
            decision_record.append(full_info)
            continue

        # otherwise generate new sentence by omission
        changes = []
        for ext in ext_version:
            if ext.tag not in ("weak", "strong"):
                continue
            triple = ext.triple[0]
            with bot as bot_ses:
                bot_ses.set_num_answers(1)
                bot_ses.set_deterministic(True)
                topic = target.replace("Please tell me about ", "")
                alt_sent = bot_ses.ask(
                    f"""
Here is the start of a description about {topic}:
{ext.prefix}

Original Sentence: 
{alt_sent}

This sentence originally followed the description. However, there is a contradiction with this sentence:
{ext.alt}

Remove the conflicting information from the sentence, preserving as much valid information as possible.
The result must fit well after the given description. Answer with the new sentence only.
The new sentence must only contain information from the original sentence."""
                )[0].strip()
            changes.append((triple, ext.alt, alt_sent))
        new_desc.append(alt_sent)
        full_info["decision"] = "redact" if changes else "keep"
        full_info["changes"] = changes
        full_info["sent_new"] = alt_sent
        decision_record.append(full_info)
    desc_new = " ".join(new_desc)
    return desc_new, decision_record


def rephrase_sent_remove_conflict(
    bot,
    sent,
    alt_sent,
    target,
    prefix,
):
    """Asks the model to remove information from a sentence that was deemed self-contradictory"""

    # otherwise generate new sentence by omission
    with bot as bot_ses:
        bot_ses.set_num_answers(1)
        bot_ses.set_deterministic(True)
        alt_sent = bot_ses.ask(
            f"""
This is the start of a text answering the prompt "{target}":
{prefix}

This is the original next sentence: 
{sent}

However, there is a contradiction with the following sentence:
{alt_sent}

Remove the conflicting information from the original sentence, preserving as much non-conflicting information as possible.
The result must fit well after the given description, it should not copy content of the description. Answer with the new sentence only.
The new sentence must only contain information from the original sentence.
If there is no non-conflicting information, produce an empty string.
"""
        )
    return alt_sent
