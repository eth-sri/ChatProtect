import dataclasses
import json
import pathlib
import re
import typing
import os
import logging

import secret
from chatprotect.bots import (
    openai_chat_api,
    openai_completion_api,
    fastchat_api,
    together_ai_api,
)
from secret import OPENAI_API_KEY

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)

from dataclasses import dataclass

import nltk

# nltk.download("punkt")
from nltk import sent_tokenize


def split_sentences(text):
    return sent_tokenize(text)


CHECKED = 0
INCONSISTENT = 1
SPURIOUS = -1

EXAMPLES_DESC_INCONSISTENCIES = [
    (
        "Angela Merkel is a German politician who served as the Chancellor of Germany from 2005 to 2021. She is considered one of the most powerful women in the world and has been a leading figure in European politics for over a decade. Merkel was born in Hamburg in May 1954 and grew up in East Germany. She studied physics at the University of Leipzig and earned a doctorate in quantum chemistry.",
        """Angela Merkel is a German politician who served as the Chancellor of Germany from 2005 to 2021. She was born on May 17, 1954, in Hamburg, West Germany. Merkel is the first woman to hold the position of Chancellor in Germany and has been described as one of the most powerful women in the world.
Merkel was educated at the University of Berlin, where she studied physics. She later earned a doctorate in quantum chemistry. After the fall of the Berlin Wall, Merkel became involved in politics and joined the Christian Democratic Union (CDU) party in 1990.""",
        (
            "The place of education is different in the two descriptions. The first description states that Merkel studied at the University of Leipzig, while the seondd description states she was educated at the University of Berlin",
            "She studied physics at the University of Leipzig",
            "Merkel was educated at the University of Berlin",
        ),
    ),
    (
        "Hernán Siles Zuazo (1914-1996) was a Bolivian politician who served as the President of Bolivia on three occasions, from 1956 to 1960, 1982 to 1985, and 1985 to 1989. He was a member of the Revolutionary Nationalist Movement (MNR) party, which he helped found in 1942. He spent his whole life in Bolivia.",
        "Hernán Siles Zuazo was a Bolivian politician who served as President of Bolivia on two separate occasions. He was born on March 21, 1914, in La Paz, Bolivia, and died on August 6, 1996, in Bethesda, Maryland, USA.",
        (
            "In the two descriptions Hernán Siles Zuazo served as president for a different number of times. In description 1, he served on three occasions, while description 2 states that he served two times.",
            "Hernán Siles Zuazo (1914-1996) was a Bolivian politician who served as the President of Bolivia on three occasions",
            "Hernán Siles Zuazo was a Bolivian politician who served as President of Bolivia on two separate occasions.",
        ),
        (
            "In description 1, Hernán Siles Zuazo died in Bethesda, Maryland, USA, while he spent his whole life in Bolivia according to the second description. This is contradictory because he can not have died in the USA when he was in Bolivia his whole life.",
            "He was born on March 21, 1914, in La Paz, Bolivia, and died on August 6, 1996, in Bethesda, Maryland, USA.",
            "He spent his whole life in Bolivia.",
        ),
    ),
    (
        "Italy is a country located in the south of Europe and borders Slovenia, Austria, Switzerland and France.",
        "Italy is a southern European country, famous for its wine and fine foods.",
    ),
]


@dataclass(unsafe_hash=True)
class ExtSentence:
    target: str
    orig: str
    alt: str
    prefix: str
    triple: typing.Tuple[str, str, str]
    tag: typing.Union[
        typing.Literal["ok"], typing.Literal["weak"], typing.Literal["strong"]
    ]
    wrong: str = "unk"
    orig_tag: typing.Optional[
        typing.Union[
            typing.Literal["ok"], typing.Literal["weak"], typing.Literal["strong"]
        ]
    ] = None
    explanation: typing.Optional[str] = None


@dataclasses.dataclass
class SentCost:
    prompt: str
    cost_user: typing.Optional[int]
    cost_model: typing.Optional[int]


@dataclasses.dataclass
class Decisions:
    prompt: str
    decisions: list


def read_entities(path: typing.Union[pathlib.Path, str]):
    topics = []
    with pathlib.Path(path).open() as f:
        for l in f:
            entity = l.strip()
            # this is a category
            if entity.startswith("[") or entity.startswith("#"):
                continue
            if not entity:
                continue
            topics.append(entity)
    return topics


def read_sent_file(target: typing.Optional[str], path: typing.Union[pathlib.Path, str]):
    if "_cost" in str(path):
        raise ValueError("trying to read cost file")
    with pathlib.Path(path).open("r") as f:
        d = json.load(f)
    ext = ExtSentence(**d)
    if target is not None and ext.target != target:
        raise ValueError("Invalid sents file for given target")
    return ext


def read_sent_file_old(
    target: typing.Optional[str], path: typing.Union[pathlib.Path, str]
):
    if "_cost" in str(path):
        raise ValueError("trying to read cost file")
    with pathlib.Path(path).open("r") as f:
        test_target = f.readline().strip()
        if target is not None and test_target != target:
            raise ValueError("Invalid sents file for given target")
        tag = f.readline().strip()  # ok, weak or strong
        wrong = f.readline().strip()  # none, 1, 2 or both
        aux = eval(f.readline().strip())  # ('S' , 'P' , 'O')
        # skip origin
        # f.readline()
        f.readline()
        cont = f.readline()
        # read prefix
        prefix = ""
        while cont != "---;;;\n":
            prefix += cont
            cont = f.readline()
        # read explanation
        cont = f.readline()
        exp = ""
        while cont != "---;;;\n":
            exp += cont
            cont = f.readline()
        # read sent 1
        s1 = ""
        cont = f.readline()
        exp_present = True
        while cont != "---;;;\n":
            s1 += cont
            cont = f.readline()
            if not cont:
                exp_present = False
                break
        if exp_present:
            # read sent 2
            s2 = f.read()
        else:
            s2 = s1
            s1 = exp
            exp = None
        prefix = prefix.strip()
        exp = exp.strip() if exp is not None else None
        s1 = s1.strip()
        s2 = s2.strip()
    return ExtSentence(test_target, s1, s2, prefix, aux, tag, wrong, exp)


@dataclasses.dataclass
class Description:
    prompt: str
    text: str
    cost_user: typing.Optional[int]
    cost_model: typing.Optional[int]


def read_desc_file_old(
    target: typing.Optional[str], path: typing.Union[pathlib.Path, str]
):
    with pathlib.Path(path).open("r") as f:
        test_target = f.readline().strip()
        if target is not None and test_target != target:
            raise ValueError("invalid desc file for target")
        # skip the two lines of cost
        try:
            cost_user = int(f.readline())
            cost_model = int(f.readline())
            # skip one empty line
            f.readline()
        except ValueError:
            cost_user = cost_model = None
        # read prefix
        desc = f.read().strip()
    return Description(test_target, desc, cost_user, cost_model)


def read_desc_file(target: typing.Optional[str], path: typing.Union[pathlib.Path, str]):
    with pathlib.Path(path).open("r") as f:
        d = json.load(f)
    desc = Description(**d)
    test_target = desc.prompt
    if target is not None and test_target != target:
        raise ValueError("invalid desc file for target")
    return desc


def label_to_tag(label: int) -> str:
    if label == INCONSISTENT:
        tag = "strong"
    elif label == CHECKED:
        tag = "ok"
    else:
        tag = "ok"
    return tag


@dataclasses.dataclass
class State:
    sentences: typing.List[str] = dataclasses.field(default_factory=list)
    current_sentence: typing.Optional[int] = None
    alternative: typing.Optional[str] = None
    contradiction: typing.Optional[bool] = None
    mitigation: typing.Optional[str] = None
    generating: typing.Literal[
        "sentences", "alternative", "contradiction", "mitigation"
    ] = "sentences"
    status: typing.Literal["starting", "producing", "done"] = "starting"
    decisions: typing.Dict[str, typing.List] = dataclasses.field(default_factory=dict)
    step: int = 0
    explanation: typing.Optional[str] = None
    decision: typing.Literal["keep", "drop", "redact"] = None
    triple: typing.List[str] = dataclasses.field(default_factory=lambda: ["", "", ""])

    def to_json(self):
        return json.dumps(dataclasses.asdict(self)) + "\n"


def fetch_model(model):
    if model in ("chatgpt", "gpt4"):
        bot = openai_chat_api.OpenAIBot(OPENAI_API_KEY, model)
    elif model in ("davinci", "babbage", "ada", "text-davinci-003"):
        bot = openai_completion_api.OpenAIBot(OPENAI_API_KEY, model)
    elif model in (f"Llama-2-{i}b-chat-hf" for i in [7, 13, 70]):
        bot = openai_chat_api.OpenAIBot(
            "EMPTY", model, openai_baseurl=openai_chat_api.FASTCHAT_OPENAI_BASE
        )
    elif model in (f"llama-2-{i}b-chat" for i in [7, 13, 70]):
        bot = together_ai_api.TogetherAIBot(
            secret.TOGETHER_API_KEY,
            model,
        )
    else:
        bot = fastchat_api.FastChatBot(model=model)
    return bot


def prompt_identifier(x: str):
    x = re.sub(r"\s", "_", x)
    x = re.sub(r"[^a-zA-Z0-9_.]", "", x)
    return x[:256]
