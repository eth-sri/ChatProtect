import dataclasses
import json
import typing
import datetime
import time
from random import random
import logging

import requests

from .bot import Bot

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass()
class Usage:
    prompt_tokens: typing.List[int]
    completion_tokens: typing.List[int]


# source: https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5#64b8e6cdf8bf823a61ed1243
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"


def llama_v2_prompt(messages: typing.List[dict], seed=None):
    if seed is not None:
        seed_string = f"(seed={seed}) "
    else:
        seed_string = ""
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {seed_string}{(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


class TogetherAIBot(Bot):
    def __init__(
        self, api_key, model, url="https://api.together.xyz/inference", seed=None
    ):
        self.api_key = api_key
        self.model = model
        self.url = url
        self.user_agent = "chatprotect"
        self.last_request = datetime.datetime.now()
        self.total_usage = []
        self.stream = False
        self.seed = seed

    def _ask(
        self,
        prompt: str,
        system_prompt=None,
        system_hist=(),
        history=(),
        num=1,
        deterministic=False,
        stop_seq=None,
        override_temperature=None,
    ):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        for question, answer in system_hist:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        for question, answer in history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        if prompt:
            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )
        full_prompt = llama_v2_prompt(messages, seed=self.seed)
        full_model = f"togethercomputer/{self.model}"
        body = {
            "model": full_model,
            "prompt": full_prompt,
            "max_tokens": 2048,
            "stop": [
                B_INST,
                B_SYS,
                BOS,
            ],
            "temperature": (
                0.99 if override_temperature is None else override_temperature
            ),
        }
        total_choices = []
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        while len(total_choices) < num:
            res_json = None
            while res_json is None:
                for i in range(100):
                    try:
                        res_raw = requests.post(
                            self.url,
                            json=body,
                            headers=headers,
                            timeout=datetime.timedelta(minutes=5).total_seconds(),
                        ).text
                        res_json = json.loads(res_raw)
                    except Exception as e:
                        _LOGGER.warning(e)
                        # exponential backoff
                        time.sleep((1 + random()) ** i)
                    else:
                        break
            choices = [a["text"] for a in res_json["output"]["choices"]]
            self.last_request = datetime.datetime.now()
            for a in choices:
                _LOGGER.debug("A: " + a)
            print(json.dumps({"Q": prompt, "A": choices}))
            total_choices.extend(choices)
        return total_choices, Usage(0, 0)
