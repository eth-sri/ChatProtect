import dataclasses
import datetime
import json
import time
import logging
from collections import defaultdict
from typing import Dict

from .bot import Bot, Usage

import tiktoken

# Daily cost limit in USD
DAILY_LIMIT = 50
TOKEN_INPUT_COST_MAP = {
    "gpt-3.5-turbo-0301": 0.0015 / 1000,
    "gpt-4-0314": 0.03 / 1000,
}
TOKEN_OUTPUT_COST_MAP = {
    "gpt-3.5-turbo-0301": 0.002 / 1000,
    "gpt-4-0314": 0.06 / 1000,
}


_LOGGER = logging.getLogger(__name__)

MODEL_MAP = {"chatgpt": "gpt-3.5-turbo-0301", "gpt4": "gpt-4-0314"}
ENCODING_MAP = {m: tiktoken.encoding_for_model(m) for m in MODEL_MAP.values()}

FASTCHAT_OPENAI_BASE = "http://localhost:8000/v1"


def to_hour(dt: datetime.datetime) -> int:
    return int(dt.timestamp()) // 3600


@dataclasses.dataclass
class CostMap:

    cost_map: Dict[str, Dict[int, int]] = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    def add_cost(self, model: str, tokens: int):
        """Add cost incurred to the cost map"""
        self.cost_map[model][to_hour(datetime.datetime.now())] += tokens

    def get_cost(self, model: str) -> int:
        """Get cost incurred by the model in the last 24 hours"""
        # this also cleans up the cost map
        earliest_admissable_hour = to_hour(datetime.datetime.now()) - 24
        too_early_hours = []
        sum = 0
        for hour, tokens in self.cost_map[model].items():
            if hour < earliest_admissable_hour:
                too_early_hours.append(hour)
            else:
                sum += tokens
        for hour in too_early_hours:
            del self.cost_map[model][hour]
        return sum


prompt_tokens = CostMap()
completion_tokens = CostMap()


def daily_limit_exceeded() -> bool:
    total_cost = 0
    for model in MODEL_MAP.values():
        total_cost += prompt_tokens.get_cost(model) * TOKEN_INPUT_COST_MAP[model]
        total_cost += completion_tokens.get_cost(model) * TOKEN_OUTPUT_COST_MAP[model]
    return total_cost >= DAILY_LIMIT


@dataclasses.dataclass
class Cost:
    prompt_tokens: int
    completion_tokens: int


class OpenAIBot(Bot):
    def __init__(self, api_key, model="chatgpt", openai_baseurl=None):
        import openai

        openai.api_key = api_key
        if openai_baseurl is not None:
            openai.api_base = openai_baseurl
        self.bot = openai.ChatCompletion()
        if openai_baseurl is None:
            self.model = MODEL_MAP[model]
        else:
            self.model = model
        self.last_request = datetime.datetime.now()

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
            messages.append(
                {"role": "system", "name": "example_user", "content": question}
            )
            messages.append(
                {"role": "system", "name": "example_assistant", "content": answer}
            )
        for question, answer in history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if daily_limit_exceeded():

            def error_stream():
                choices = [
                    "Daily token limit exceeded. We only have a finite budget of OpenAI API tokens for this website. "
                    "If you find our tool helpful and wish to continue using it, please run our code with your own API keys."
                ] * num
                yield choices

            return error_stream(), Cost(0, 0)
        for i in range(3):
            try:
                results = self.bot.create(
                    model=self.model,
                    messages=messages,
                    temperature=(1 if not deterministic else 0)
                    if override_temperature is None
                    else override_temperature,
                    n=num,
                    stream=True,
                )
                prompt_tokens.add_cost(
                    self.model,
                    sum(
                        len(ENCODING_MAP[self.model].encode(m["content"]))
                        for m in messages
                    ),
                )

                def stream():
                    choices = [""] * num
                    for res in results:
                        deltas = [(a.index, a.delta) for a in res.choices]
                        for i, delta in deltas:
                            if "content" in delta:
                                choices[i] += delta.content
                        self.last_request = datetime.datetime.now()
                        for a in choices:
                            _LOGGER.debug("A: " + a)
                        yield choices
                    completion_tokens.add_cost(
                        self.model,
                        sum(len(ENCODING_MAP[self.model].encode(c)) for c in choices),
                    )

                return stream(), Cost(0, 0)
            except Exception as e:
                _LOGGER.warning(e)
                # exponential backoff
                time.sleep(2**i)

        def error_stream():
            choices = [
                "Sorry, there was an issue connecting to the OpenAI API. Please try again later."
            ] * num
            yield choices

        return error_stream(), Cost(0, 0)
