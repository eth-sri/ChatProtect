import datetime
import json
import time
import logging

from .bot import Bot, Usage

_LOGGER = logging.getLogger(__name__)

MODEL_MAP = {"chatgpt": "gpt-3.5-turbo-0301", "gpt4": "gpt-4-0314"}
PROXY_MODEL_MAP = {
    "llama3.1-8b-instruct": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-8b-instruct": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "gemma3-12b-instruct": "openrouter/google/gemma-3-12b-it",
    "gemma-3-12b-it": "openrouter/google/gemma-3-12b-it",
}

FASTCHAT_OPENAI_BASE = "http://localhost:8000/v1"


class OpenAIBot(Bot):
    def __init__(self, api_key, model="chatgpt", openai_baseurl=None):
        import openai

        openai.api_key = api_key
        if openai_baseurl is not None:
            openai.api_base = openai_baseurl
        self.bot = openai.ChatCompletion()
        self.model = MODEL_MAP.get(model, PROXY_MODEL_MAP.get(model, model))
        self.last_request = datetime.datetime.now()
        self.total_usage = []

    @staticmethod
    def _parse_usage(res):
        usage = getattr(res, "usage", None)
        if usage is None:
            return Usage(0, 0)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        return Usage(prompt_tokens or 0, completion_tokens or 0)

    def _ask(
        self,
        prompt: str,
        system_prompt=None,
        system_hist=(),
        system_prompt_role="system",
        history=(),
        num=1,
        deterministic=False,
        stop_seq=None,
        override_temperature=None,
        response_format=None,
        provider=None,
        top_k=None,
        top_p=None,
    ):
        messages = []
        if system_prompt is not None:
            messages.append({"role": system_prompt_role, "content": system_prompt})
        for question, answer in system_hist:
            if system_prompt_role == "system":
                messages.append(
                    {"role": "system", "name": "example_user", "content": question}
                )
                messages.append(
                    {
                        "role": "system",
                        "name": "example_assistant",
                        "content": answer,
                    }
                )
            else:
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
        for question, answer in history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        if prompt:
            messages.append({"role": "user", "content": prompt})
        for i in range(1000):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=(1 if not deterministic else 0)
                    if override_temperature is None
                    else override_temperature,
                    n=num,
                    stream=False,
                    response_format=response_format,
                )
                if provider is not None:
                    kwargs["provider"] = provider
                if top_k is not None:
                    kwargs["top_k"] = top_k
                if top_p is not None:
                    kwargs["top_p"] = top_p
                res = self.bot.create(**kwargs)
                choices = res.choices
                if choices is None:
                    raise ValueError(f"API returned choices=None: {res}")
                usage = self._parse_usage(res)
                self.total_usage.append(usage)
                break
            except Exception as e:
                _LOGGER.warning(e)
                # exponential backoff
                time.sleep(2**i)
        choices = [a.message.content for a in choices]
        print(json.dumps({"Q": prompt, "A": choices}))
        self.last_request = datetime.datetime.now()
        _LOGGER.debug(json.dumps({"Q": prompt, "A": choices}))
        return choices, usage
