import re
from typing import Dict, List, Any

from chatprotect.bots.bot import Bot
from chatprotect.util import CHECKED, INCONSISTENT, ExtSentence


def check_consistent_ask_direct(
    bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str
):
    """Ask the model directly if it finds a contradiction between the sentences"""
    ask_prompt = f"""\
I give you the beginning of a description about {target}.
Then follow two statements.

Description:
{prefix}

Statement 1:
{stmt1}

Statement 2:
{stmt2}

Are the two statements about {target} contradictory? Answer with either Yes or No.
"""
    with bot as bot_ses:
        bot_ses.set_num_answers(10)
        bot_ses.set_deterministic(False)
        ress_raw = bot_ses.ask(ask_prompt)
    ress = []
    for res in ress_raw:
        if re.findall(r"\bno\b", res.lower()):
            ress.append(CHECKED)
        elif not re.findall(r"\byes\b", res.lower()):
            # spurious... is ok though
            ress.append(CHECKED)
        else:
            ress.append(INCONSISTENT)
    # the score is the likelihood of returning yes/no
    return sum(ress) / len(ress)


def explain_consistent_cot(bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str):
    """Ask the model if it finds a contradiction between the sentences. Uses Chain of Thought to improve the result"""
    explain_prompt = f"""\
I give you the beginning of a text answering the prompt "{target}".
Then follow two statements.

Text:
{prefix}

Statement 1:
{stmt1}

Statement 2:
{stmt2}

Please explain whether the statements about {target} are contradictory.
Provide your explanation only.
"""
    with bot as bot_ses:
        bot_ses.set_deterministic(True)
        bot_ses.set_num_answers(1)
        res = bot_ses.ask(explain_prompt)
    return res


def explain_consistent_cot_original(
    bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str
):
    """Ask the model if it finds a contradiction between the sentences. Uses Chain of Thought to improve the result"""
    topic = target.replace("Please tell me about ", "")
    explain_prompt = f"""\
I give you the beginning of a description about {topic}.
Then follow two statements.

Description:
{prefix}

Statement 1:
{stmt1}

Statement 2:
{stmt2}

Please explain whether the statements about {topic} are contradictory.
Provide your explanation only.
"""
    with bot as bot_ses:
        bot_ses.set_deterministic(True)
        bot_ses.set_num_answers(1)
        res = bot_ses.ask(explain_prompt)
    return res


def check_consistent_cot_original(
    bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str, reason: str
):
    """Ask the model if it finds a contradiction between the sentences. Uses Chain of Thought to improve the result"""
    if stmt1 == stmt2:
        return CHECKED
    topic = target.replace("Please tell me about ", "")
    explain_prompt = f"""\
I give you the beginning of a description about {topic}.
Then follow two statements.

Description:
{prefix}

Statement 1:
{stmt1}

Statement 2:
{stmt2}

Please explain whether the statements about {topic} are contradictory.
Provide your explanation only.
"""
    with bot as bot_ses:
        bot_ses.add_history(explain_prompt, reason)
        bot_ses.set_deterministic(False)
        bot_ses.set_num_answers(10)
        conclusions_raw = bot_ses.ask(
            "Please conclude whether the statements are contradictory with Yes or No."
        )
    conclusions = []
    for conclusion in conclusions_raw:
        follows_yes = re.findall(r"\byes\b", conclusion.lower())
        follows_no = re.findall(r"\bno\b", conclusion.lower())
        if follows_yes and not follows_no:
            conclusions.append(INCONSISTENT)
        elif follows_no and not follows_yes:
            conclusions.append(CHECKED)
        else:
            # spurious... is ok though
            conclusions.append(CHECKED)
    return sum(conclusions) / len(conclusions)


def check_consistent_cot(
    bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str, reason: str
):
    """Ask the model if it finds a contradiction between the sentences. Uses Chain of Thought to improve the result"""
    if stmt1 == stmt2:
        return CHECKED
    explain_prompt = f"""\
I give you the beginning of a text answering the prompt "{target}".
Then follow two statements.

Text:
{prefix}

Statement 1:
{stmt1}

Statement 2:
{stmt2}

Please explain whether the statements about {target} are contradictory.
Provide your explanation only.
"""
    with bot as bot_ses:
        bot_ses.add_history(explain_prompt, reason)
        bot_ses.set_deterministic(False)
        bot_ses.set_num_answers(10)
        conclusions_raw = bot_ses.ask(
            "Please conclude whether the statements are contradictory with Yes or No."
        )
    conclusions = []
    for conclusion in conclusions_raw:
        follows_yes = re.findall(r"\byes\b", conclusion.lower())
        follows_no = re.findall(r"\bno\b", conclusion.lower())
        if follows_yes and not follows_no:
            conclusions.append(INCONSISTENT)
        elif follows_no and not follows_yes:
            conclusions.append(CHECKED)
        else:
            # spurious... is ok though
            conclusions.append(CHECKED)
    return sum(conclusions) / len(conclusions)


def check_consistent_cot_stream(
    bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str, reason: str
):
    """Ask the model if it finds a contradiction between the sentences. Uses Chain of Thought to improve the result"""
    if stmt1 == stmt2:
        return CHECKED
    explain_prompt = f"""\
I give you the beginning of a text answering the prompt "{target}".
Then follow two statements.

Text:
{prefix}

Statement 1:
{stmt1}

Statement 2:
{stmt2}

Please explain whether the statements about {target} are contradictory.
Provide your explanation only.
"""
    with bot as bot_ses:
        bot_ses.add_history(explain_prompt, reason)
        bot_ses.set_deterministic(False)
        bot_ses.set_num_answers(10)
        conclusions_raw_generator = bot_ses.ask(
            "Please conclude whether the statements are contradictory with Yes or No."
        )
    for conclusions_raw in conclusions_raw_generator:
        pass
    conclusions = []
    for conclusion in conclusions_raw:
        follows_yes = re.findall(r"\byes\b", conclusion.lower())
        follows_no = re.findall(r"\bno\b", conclusion.lower())
        if follows_yes and not follows_no:
            conclusions.append(INCONSISTENT)
        elif follows_no and not follows_yes:
            conclusions.append(CHECKED)
        else:
            # spurious... is ok though
            conclusions.append(CHECKED)
    return sum(conclusions) / len(conclusions)


def check_nonfactual_cot(bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str):
    """Ask the model if it thinks the first sentences is nonfactual. Uses Chain of Thought to improve the result"""
    explain_prompt = f"""\
I give you the beginning of a description about {target}.
Then follows one statement.

Description:
{prefix}

Statement:
{stmt1}

Please explain whether the statement about {target} is nonfactual.
Provide your explanation only.
"""
    with bot as bot_ses:
        bot_ses.set_deterministic(True)
        bot_ses.set_num_answers(1)
        res = bot_ses.ask(explain_prompt)[0]
        bot_ses.add_history(explain_prompt, res)
        bot_ses.set_deterministic(False)
        bot_ses.set_num_answers(10)
        conclusions_raw = bot_ses.ask(
            "Please conclude whether the statement is nonfactual with Yes or No."
        )
    conclusions = []
    for conclusion in conclusions_raw:
        follows_yes = re.findall(r"\byes\b", conclusion.lower())
        follows_no = re.findall(r"\bno\b", conclusion.lower())
        if follows_yes and not follows_no:
            conclusions.append(INCONSISTENT)
        elif follows_no and not follows_yes:
            conclusions.append(CHECKED)
        else:
            # spurious... is ok though
            conclusions.append(CHECKED)
    return sum(conclusions) / len(conclusions)


def check_consistent_cot_multivote(
    bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str
):
    """Ask the model if it thinks the sentences are contradictory. Uses Chain of Thought with 5 explanations to improve the result"""
    if stmt1 == stmt2:
        return CHECKED
    explain_prompt = f"""\
I give you the beginning of a description about {target}.
Then follow two sentences.

Description:
{prefix}

Statement 1:
{stmt1}

Statement 2:
{stmt2}

Please explain whether the statements about {target} are contradictory.
Provide your explanation only.
"""
    with bot as bot_ses:
        bot_ses.set_deterministic(False)
        bot_ses.set_num_answers(5)
        results = bot_ses.ask(explain_prompt)
    conclusions_raw = []
    for res in results:
        with bot as bot_ses:
            bot_ses.set_deterministic(True)
            bot_ses.set_num_answers(10)
            bot_ses.add_history(explain_prompt, res)
            conclusion = bot_ses.ask(
                "Please conclude whether the statements are contradictory. Answer with Yes or No."
            )
            conclusions_raw.append(conclusion)
    inc, approve, spurious = 0, 0, 0
    conclusions = []
    for conclusion_list in conclusions_raw:
        for conclusion in conclusion_list:
            follows_yes = re.findall(r"\byes\b", conclusion.lower())
            follows_no = re.findall(r"\bno\b", conclusion.lower())
            if follows_yes and not follows_no:
                inc += 1
            elif follows_no and not follows_yes:
                approve += 1
            else:
                spurious += 1
        # return majority vote or spurious if unclear result
        total = inc + approve + spurious
        conclusions.append(inc / total)
    return sum(conclusions) / len(conclusions)


def check_consistent_score(bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str):
    """Ask the model if it thinks the sentences are contradictory. Uses Chain of Thought to improve the result. Returns a score between 0-10 instead of Yes/No."""
    if stmt1 == stmt2:
        return CHECKED
    explain_prompt = f"""\
I give you the beginning of a text answering the prompt "{target}".
Then follow two statements.

Text:
{prefix}

Statement 1:
{stmt1}

Statement 2:
{stmt2}

Please explain whether the statements about {target} are contradictory.
Provide your explanation only.
"""
    with bot as bot_ses:
        bot_ses.set_deterministic(True)
        bot_ses.set_num_answers(1)
        res = bot_ses.ask(explain_prompt)[0]
        bot_ses.add_history(explain_prompt, res)
        bot_ses.set_deterministic(False)
        bot_ses.set_num_answers(10)
        conclusions_raw = bot_ses.ask(
            "Please conclude whether the statements are contradictory with a score between 0 (no contradiction) and 10 (strong contradiction)."
        )
    scores = []
    for conclusion in conclusions_raw:
        number = extract_score(conclusion)
        scores.append(number / 10)
    return sum(scores) / len(scores) if scores else 0


def check_factual_multi_score(
    bot: Bot, stmt1: str, stmt2s: List[str], target: str, prefix: str
):
    """Ask the model if it thinks the sentences are contradictory. Uses Chain of Thought to improve the result. Returns a score between 0-10 instead of Yes/No."""
    add_stms = "".join(
        f"""

Evidence {i}:
{stmt2}
"""
        for i, stmt2 in enumerate(stmt2s, start=1)
    )
    explain_prompt = f"""\
I give you the beginning of a text answering the prompt "{target}".
Then follow several statements. The first statement is the original statement of the text.
The subsequent statements are additional evidence for you to verify the correctness of the original statement.
If there is a contradiction between the original text and any of the additional evidences,
you should conclude that the original statement is incorrect.

Text:
{prefix}

Original Statement:
{stmt1}{add_stms}

Are there contradictions between the original statement and the provided evidence?
Based on your reasoning about the above question and the evidences, what is your conclusion regarding the correctness of the original statement?
Provide your explanation only.
"""
    with bot as bot_ses:
        bot_ses.set_deterministic(True)
        bot_ses.set_num_answers(1)
        res = bot_ses.ask(explain_prompt)[0]
        bot_ses.add_history(explain_prompt, res)
        bot_ses.set_deterministic(False)
        bot_ses.set_num_answers(10)
        conclusions_raw = bot_ses.ask(
            'Please conclude whether the statement is incorrect a score between 0 (entirely incorrect) and 10 (fully correct).\nAnswer just "Score: X" where X is your score'
        )
    scores = []
    for conclusion in conclusions_raw:
        number = extract_score(conclusion)
        scores.append(number / 10)
    return (1 - sum(scores) / len(scores)) if scores else 0


def check_consistent_step_by_step(
    bot: Bot, stmt1: str, stmt2: str, target: str, prefix: str
):
    """Ask the model if it thinks the sentences are contradictory. Uses Step by step reasoning to improve the result. Returns a score between 0-10 instead of Yes/No."""
    ask_prompt = f"""\
I give you the beginning of a description about {target}.
Then follow two statements.

Description:
{prefix}

Statement 1:
{stmt1}

Statement 2:
{stmt2}

Are the two statements about {target} contradictory? First, show your reasoning in a step-by-step fashion. Then conclude with yes or no.
"""
    with bot as bot_ses:
        bot_ses.set_num_answers(10)
        bot_ses.set_deterministic(False)
        ress_raw = bot_ses.ask(ask_prompt)
    ress = []
    for res in ress_raw:
        if re.findall(r"\bno\b", res.lower()):
            ress.append(CHECKED)
        elif not re.findall(r"\byes\b", res.lower()):
            # spurious... is ok though
            ress.append(CHECKED)
        else:
            ress.append(INCONSISTENT)
    # the score is the likelihood of returning yes/no
    return sum(ress) / len(ress)


def extract_score(conclusion: str):
    conclusion.replace("0 to 10", "")
    numbers = list(
        re.finditer(
            r"(?:(?:would|will|) score |score the statement |rate it |the score (?:is|would be|will be) |statement is |as a |as |a |score of |: ?|at )(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)(?: out of \d+|/\d+)",
            conclusion,
        )
    )
    numbers = [float(number.group(1) or number.group(2)) for number in numbers]
    numbers = [number for number in numbers if 0 <= number <= 10]
    if len(numbers) != 1:
        print(numbers)
        print(conclusion)
    return sum(numbers) / len(numbers) if numbers else None
