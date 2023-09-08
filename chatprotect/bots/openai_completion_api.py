import datetime
import time
import logging

from .bot import Bot, Usage

_LOGGER = logging.getLogger(__name__)


class OpenAIBot(Bot):
    def __init__(self, api_key, model="davinci", stream=False):
        import openai

        openai.api_key = api_key
        self.bot = openai.Completion()
        self.model = model
        self.last_request = datetime.datetime.now()
        self.total_usage = []
        self.stream = stream

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
        for i in range(1000):
            try:
                prompt = "\n###\n".join(
                    (
                        j
                        for j in (
                            "\n###\n".join("\n###\n".join(h) for h in history),
                            "\n###\n".join("\n###\n".join(h) for h in system_hist),
                            prompt,
                        )
                        if j
                    )
                )
                results = self.bot.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=(1 if not deterministic else 0)
                    if override_temperature is None
                    else override_temperature,
                    n=num,
                    stream=self.stream,
                    max_tokens=256,
                    stop=stop_seq,
                )
                if self.stream:

                    def stream():
                        choices = [""] * num
                        for res in results:
                            deltas = [a.delta for a in res.choices]
                            for i, delta in enumerate(deltas):
                                if "content" in delta:
                                    choices[i] += delta.content
                            self.last_request = datetime.datetime.now()
                            for a in choices:
                                _LOGGER.debug("A: " + a)
                            yield choices

                    return stream()
                else:
                    res = results
                choices = res.choices
                usage = res.usage
                usage = Usage(usage.prompt_tokens, usage.completion_tokens)
                self.total_usage.append(usage)
                break
            except Exception as e:
                _LOGGER.warning(e)
                # exponential backoff
                time.sleep(2**i)
        choices = [a.text for a in choices]
        self.last_request = datetime.datetime.now()
        for a in choices:
            _LOGGER.debug("A: " + a)
        return choices, usage
