import dataclasses
from typing import List, Tuple

import typing


@dataclasses.dataclass()
class Usage:
    prompt_tokens: int
    completion_tokens: int


class Bot:
    def __enter__(self):
        """Starts a chat session with the bot"""
        return BotSession(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ends the session with the bot"""
        pass

    def _ask(
        self,
        prompt: str,
        system_prompt=None,
        system_hist=(),
        history=(),
        num_answers=1,
        deterministic=False,
        stop_seq=None,
        override_temperature=None,
    ) -> Tuple[List[str], Usage]:
        raise NotImplementedError()


class BotSession:
    def __init__(self, bot: Bot):
        self.bot = bot
        self.system_prompt = None
        self.system_hist = []
        self.history = []
        self.num_answers = 1
        self.usage = Usage(0, 0)
        self.deterministic = True
        self.stop_seq = None
        self.override_temperature = None

    def set_num_answers(self, num: int):
        # can be accomplished by other systems by repeating the questions
        self.num_answers = num

    def set_stop_seq(self, seq):
        # can be accomplished by other systems by repeating the questions
        self.stop_seq = seq

    def set_deterministic(self, det: bool):
        # can be accomplished by other systems by repeating the questions
        self.deterministic = det

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def add_system_history(self, prompt, answer):
        # needs to be manually added! because we don't know which answer was chosen by the user
        self.system_hist.append((prompt, answer))

    def add_history(self, prompt, answer):
        # needs to be manually added! because we don't know which answer was chosen by the user
        self.history.append((prompt, answer))

    def ask(self, prompt: str):
        """Prompts a question to the bot and returns the result"""
        res = self.bot._ask(
            prompt,
            self.system_prompt,
            self.system_hist,
            self.history,
            self.num_answers,
            self.deterministic,
            self.stop_seq,
            self.override_temperature,
        )
        ress, cost = res
        self.usage.prompt_tokens += cost.prompt_tokens
        self.usage.completion_tokens += cost.completion_tokens
        return ress
