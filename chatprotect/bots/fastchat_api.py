import dataclasses
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


class FastChatBot(Bot):
    def __init__(self, model="opt-350m-sharegpt", url="http://localhost:8000/v1"):
        self.model = model
        self.url = url
        self.last_request = datetime.datetime.now()
        self.seed = 0
        self.total_usage = []
        self.stream = False

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
        body = {"model": self.model, "system": system_prompt}
        messages = []
        # NOTE: these roles need to fit the roles that the model was trained with
        # optimally, this is information that the model stores internally and is remapped by some API
        for question, answer in system_hist:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        _LOGGER.debug("Q: " + prompt)
        for question, answer in history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        if prompt:
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": ""})
        body["n"] = num
        body["messages"] = messages
        # NOTE: for fastchat we should always use temperature, as otherwise repetitive sentences are generated
        # this is fine regrding determinism since we are able to set a seed
        body["temperature"] = (
            1 if override_temperature is None else override_temperature
        )
        body["seed"] = self.seed
        res = None
        while res is None:
            for i in range(100):
                try:
                    res = requests.post(
                        f"{self.url}/chat/completions",
                        json=body,
                        timeout=datetime.timedelta(minutes=5).total_seconds(),
                    ).json()
                except Exception as e:
                    _LOGGER.warning(e)
                    # exponential backoff
                    time.sleep((1 + random()) ** i)
                else:
                    choices = res["choices"]
                    break
        choices = [a["message"]["content"] for a in choices]
        self.last_request = datetime.datetime.now()
        for a in choices:
            _LOGGER.debug("A: " + a)
        return choices, Usage(0, 0)
