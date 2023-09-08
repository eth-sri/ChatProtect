import json
from typing import Union

from diff_match_patch import diff_match_patch, patch_obj
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import chatprotect.bots.openai_chat_stream_api
from chatprotect.consistency import (
    check_consistent_cot,
    explain_consistent_cot,
    check_consistent_cot_stream,
)
from chatprotect.rephrase import (
    rephrase_sent_remove_conflict,
)
from chatprotect.sentences import (
    extract_triples_compact_ie,
    generate_statement_missing_object,
    generate_statement_missing_object_free,
)
from chatprotect.util import State, split_sentences
from secret import OPENAI_API_KEY

app = FastAPI()

# TODO limit to chatprotect website
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode(o: patch_obj):
    return o.__dict__


def redaction_streamer(query: str, model: str):
    prev_state = [""]

    def dumps_diff(state: State):
        dmp = diff_match_patch()
        new_state = state.to_json()
        diff = dmp.patch_make(prev_state[0], new_state)
        prev_state[0] = new_state
        return json.dumps(diff, default=encode) + "\n"

    frame = 0
    bot = chatprotect.bots.openai_chat_stream_api.OpenAIBot(
        api_key=OPENAI_API_KEY, model=model
    )
    state = State()
    state.generating = "sentences"
    state.status = "starting"
    yield dumps_diff(state)
    frame += 1

    with bot as bot_session:
        state.status = "producing"
        bot_session.set_deterministic(False)
        answer_stream = bot_session.ask(query)
    for a in answer_stream:
        m = a[0]
        state.sentences = split_sentences(m)
        yield dumps_diff(state)

    state.status = "done"
    yield dumps_diff(state)
    frame += 1

    checked = set()
    for i, sent in enumerate(state.sentences):
        for step in range(2):
            state.step = step
            if sent == "":
                continue
            if sent in checked:
                continue
            checked.add(sent)
            state.generating = "alternative"
            state.current_sentence = i
            state.alternative = ""
            state.status = "starting"
            yield dumps_diff(state)
            frame += 1

            triples = extract_triples_compact_ie(sent)
            # for the website we only choose the first triple - there will be a revision anyways
            for triple in triples:
                state.status = "producing"
                state.alternative = ""
                state.contradiction = None
                state.triple = list(triple)
                yield dumps_diff(state)
                frame += 1

                continuation_stream = generate_statement_missing_object_free(
                    bot, triple[0], triple[1], query, " ".join(state.sentences[:i])
                )
                for update in continuation_stream:
                    state.alternative = update[0]
                    yield dumps_diff(state)
                    frame += 1
                state.status = "done"
                yield dumps_diff(state)
                frame += 1
                state.contradiction = None
                state.generating = "contradiction"
                state.status = "starting"
                yield dumps_diff(state)
                frame += 1
                state.status = "producing"
                yield dumps_diff(state)
                frame += 1

                explanation_stream = explain_consistent_cot(
                    bot, sent, state.alternative, query, " ".join(state.sentences[:i])
                )
                for update in explanation_stream:
                    state.explanation = update[0]
                    yield dumps_diff(state)
                    frame += 1

                bot.stream = False
                contradiction_score = check_consistent_cot_stream(
                    bot,
                    sent,
                    state.alternative,
                    query,
                    " ".join(state.sentences[:i]),
                    state.explanation,
                )
                contradiction = contradiction_score > 0.5
                state.contradiction = contradiction
                state.status = "done"
                yield dumps_diff(state)
                frame += 1
                bot.stream = True

                state.generating = "mitigation"
                # here we fetch the decision
                state.mitigation = ""
                state.status = "starting"
                yield dumps_diff(state)
                frame += 1
                state.mitigation = ""
                state.status = "producing"
                yield dumps_diff(state)
                frame += 1
                if state.contradiction:
                    if step == 1:
                        state.decision = "drop"
                        state.mitigation = (
                            "[Hallucination removed]"
                            if len(state.sentences) > 1
                            else "The model can not provide a reliable answer for this query."
                        )
                        pass
                    elif state.contradiction:
                        checked.remove(sent)
                        state.decision = "redact"
                        mitigation_stream = rephrase_sent_remove_conflict(
                            bot,
                            sent,
                            state.alternative,
                            query,
                            " ".join(state.sentences[:i]),
                        )
                        for update in mitigation_stream:
                            state.mitigation = update[0]
                            yield dumps_diff(state)
                            frame += 1
                else:
                    state.decision = "keep"
                    state.mitigation = sent
                sent = state.mitigation
                state.status = "done"
                yield dumps_diff(state)
                frame += 1

                if f"s_{i}" not in state.decisions:
                    state.decisions[f"s_{i}"] = []
                state.decisions[f"s_{i}"].append(
                    {
                        "decision": state.decision,
                        "original": state.sentences[i],
                        "contradiction": state.contradiction,
                        "alternative": state.alternative,
                        "mitigation": state.mitigation,
                        "step": state.step,
                        "frame": frame,
                        "explanation": state.explanation,
                        "triple": state.triple,
                    }
                )
                state.sentences[i] = state.mitigation
                state.mitigation = None
                state.alternative = None
                state.contradiction = None
                state.explanation = None
                state.decision = None
                if contradiction:
                    break
        state.current_sentence = None
        state.alternative = None
    state.status = "done"
    state.generating = None
    yield dumps_diff(state)
    frame += 1


@app.get("/chat")
def stream_chat_completion(q: str, bot: Union[str, None] = "chatgpt"):
    return StreamingResponse(redaction_streamer(q, bot))
