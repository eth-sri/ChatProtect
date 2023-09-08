import datetime
import json
import time
import logging

from .bot import Bot, Usage

_LOGGER = logging.getLogger(__name__)

MODEL_MAP = {"chatgpt": "gpt-3.5-turbo-0301", "gpt4": "gpt-4-0314"}

FASTCHAT_OPENAI_BASE = "http://localhost:8000/v1"


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
        self.total_usage = []

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
        for i in range(1000):
            try:
                res = self.bot.create(
                    model=self.model,
                    messages=messages,
                    temperature=(1 if not deterministic else 0)
                    if override_temperature is None
                    else override_temperature,
                    n=num,
                    stream=False,
                )
                choices = res.choices
                usage = res.usage
                usage = Usage(usage.prompt_tokens, usage.completion_tokens)
                self.total_usage.append(usage)
                break
            except Exception as e:
                _LOGGER.warning(e)
                # exponential backoff
                time.sleep(2**i)
        choices = [a.message.content for a in choices]
        self.last_request = datetime.datetime.now()
        _LOGGER.debug(json.dumps({"Q": prompt, "A": choices}))
        return choices, usage
