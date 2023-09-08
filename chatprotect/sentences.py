import re
from typing import List

import requests

import chatprotect.bots.bot
from chatprotect.bots.bot import Bot
from chatprotect.util import split_sentences


def extract_triples_compact_ie(text: str):
    """Uses the CompactIE API to extract triples, running locally"""
    request = {"sentences": [s for s in split_sentences(text)]}
    result = requests.post("http://0.0.0.0:39881/api", json=request).json()
    return [(a["subject"], a["relation"], a["object"]) for a in result]


def generate_statement_missing_object(
    bot: Bot, subject, predicate, target, prefix, override_temperature=None, alts=1
) -> str:
    """Generates a follow up statement for the description based on a triple, resembling a cloze test."""
    topic = target.replace("Please tell me about ", "")
    if not prefix:
        prefix = "There is no preceding description."
    statement_template = """
Here is the start of a description about {}:
{}

Please generate the next sentence of this description.
The generated sentence must fill the gap in this Subject;Predicate;Object triple: ({}; {}; _)
The sentence should contain as little other information as possible.
"""
    with bot as bot_ses:
        bot_ses.set_system_prompt(
            "You are a description generator. You are given the start of an description and a question that should be answered by the next sentence. You return the next sentence for the description."
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Douglas Adams",
                """
Douglas Adams was a British author and humorist best known for his science fiction series. He was born in 1952 and began his writing career as a radio scriptwriter for the BBC. 
        """,
                "Douglas Adams",
                "most famous work is",
            ),
            'Adams most famous work is the book "The Hitchhiker\'s Guide to the Galaxy"',
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Kayne West",
                "Kanye West is a rapper, producer, fashion designer, and entrepreneur known for his controversial behavior and outspoken personality.",
                "West",
                "was most recently married to",
            ),
            "He was most recently married to Kim Kardashian.",
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Angela Merkel",
                "There is no preceding description",
                "Angela Merkel",
                "was born in the city",
            ),
            "Angela Merkel was born in Hamburg, West Germany.",
        )
        bot_ses.set_deterministic(False if override_temperature is not None else True)
        bot_ses.override_temperature = override_temperature
        bot_ses.set_num_answers(alts)
        generate_sentence = bot_ses.ask(
            statement_template.format(topic, prefix, subject, predicate),
        )
    return generate_sentence


def generate_statement_missing_object_free(
    bot: Bot, subject, predicate, target, prefix, override_temperature=None, alts=1
) -> str:
    """Generates a follow up statement for the description based on a triple, resembling a cloze test."""
    if not prefix:
        prefix = "There is no preceding answer."
    statement_template = """
Here is the start of an answer to the prompt "{}":
{}

Please generate the next sentence of this answer.
The generated sentence must fill the gap in this Subject;Predicate;Object triple: ({}; {}; _)
The sentence should contain as little other information as possible.
"""
    with bot as bot_ses:
        bot_ses.set_system_prompt(
            "You are an answer generator."
            "You are given the start of an answer and a question that should be answered by the next sentence."
            "You return the next sentence for the answer."
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Please tell me about Douglas Adams",
                """
Douglas Adams was a British author and humorist best known for his science fiction series. He was born in 1952 and began his writing career as a radio scriptwriter for the BBC. 
        """,
                "Douglas Adams",
                "most famous work is",
            ),
            'Adams most famous work is the book "The Hitchhiker\'s Guide to the Galaxy"',
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Who is Kayne West?",
                "Kanye West is a rapper, producer, fashion designer, and entrepreneur known for his controversial behavior and outspoken personality.",
                "West",
                "was most recently married to",
            ),
            "He was most recently married to Kim Kardashian.",
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Where was Angela Merkel born?",
                "There is no preceding answer",
                "Angela Merkel",
                "was born in the city",
            ),
            "Angela Merkel was born in Hamburg, West Germany.",
        )
        bot_ses.set_deterministic(False if override_temperature is not None else True)
        bot_ses.override_temperature = override_temperature
        bot_ses.set_num_answers(alts)
        generate_sentence = bot_ses.ask(
            statement_template.format(target, prefix, subject, predicate),
        )
    return generate_sentence


def generate_statement_prefix(bot: Bot, target, prefix, n: int) -> List[str]:
    """Generates a follow up statement for the description based on the preceding description only."""
    if not prefix:
        prefix = "There is no preceding description."
    statement_template = """
Here is the start of a description with the topic "{}".
{}

Please generate two valid continuations of this description.
"""
    with bot as bot_ses:
        bot_ses.set_system_prompt(
            "You are a description generator. You are given the start of an description. You return potential next sentences for the description."
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Douglas Adams",
                """
Douglas Adams was a British author and humorist best known for his science fiction series. He was born in 1952 and began his writing career as a radio scriptwriter for the BBC. 
        """,
            ),
            """
- Adams most famous work is the book "The Hitchhiker\'s Guide to the Galaxy"
- There Adams soon realized that he preferred writing short stories.
            """,
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Kayne West",
                "Kanye West is a rapper, producer, fashion designer, and entrepreneur known for his controversial behavior and outspoken personality.",
            ),
            """
 - He was most recently married to Kim Kardashian.
 - Even though he faces much critisism from the community for his behaviour, he enjoys huge econmic success with his different endeavours.
            """,
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Angela Merkel",
                "There is no preceding description",
            ),
            """
- Angela Merkel was born in Hamburg, West Germany.
- Angela Merkel is a German politician, and was the first female Chancellor of the country.
            """,
        )
        bot_ses.set_deterministic(True)
        bot_ses.set_num_answers(1)
        generate_sentence = bot_ses.ask(statement_template.format(target, prefix))[0]
    sentences = re.findall(
        r"^\s*-?\s*(\d+[.)]?)?\s*(?P<question>[^\n?]+)\n", generate_sentence, re.M
    )
    sentences = list(map(lambda x: x[1], sentences))
    return sentences


def generate_statement_prefix_rephrase(
    bot: Bot, target, sentence, prefix, n: int
) -> List[str]:
    """Generates a follow up statement for the description based on the preceding description and another follow up, asking to rephrase the follow up."""
    if not prefix:
        prefix = "There is no preceding description."
    statement_template = """
Here is the start of a description with the topic "{}".
{}

Please generate the next sentence of this description.
It should be a rephrased version of this sentence:
{}
"""
    with bot as bot_ses:
        bot_ses.set_system_prompt(
            "You are a description generator. You are given the start of an description. You return the next sentence for the description."
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Douglas Adams",
                """
Douglas Adams was a British author and humorist best known for his science fiction series. He was born in 1952 and began his writing career as a radio scriptwriter for the BBC. 
        """,
                'The book "The Hitchhiker\'s Guide to the Galaxy" is what Adams is best know for.',
            ),
            'Adams most famous work is the book "The Hitchhiker\'s Guide to the Galaxy"',
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Kayne West",
                "Kanye West is a rapper, producer, fashion designer, and entrepreneur known for his controversial behavior and outspoken personality.",
                "He last wife was Kim Kardashian.",
            ),
            "He was most recently married to Kim Kardashian.",
        )
        bot_ses.add_system_history(
            statement_template.format(
                "Angela Merkel",
                "There is no preceding description",
                "She was born in the city of Hamburg in West Germany.",
            ),
            "Angela Merkel was born in Hamburg, West Germany.",
        )
        bot_ses.set_deterministic(True)
        bot_ses.set_num_answers(1)
        generate_sentence = bot_ses.ask(
            statement_template.format(target, prefix, sentence)
        )
    return generate_sentence


def extract_open_questions(bot: Bot, target: str, sentence: str, prefix):
    kg_prompt = f"""
Here is the start of a description with the topic "{target}".
{prefix}

Please read the following sentence. Write at least two questions that can be answered by the information presented in the following sentence.
Sentence:
"{sentence}"
    """
    # TODO make more resilient
    with bot as bot_ses:
        bot_ses.set_deterministic(True)
        bot_ses.set_num_answers(1)
        kg_res = bot_ses.ask(kg_prompt)[0]
    questions = re.findall(
        r"^\s*-?\s*(\d+[.)]?)?\s*(?P<question>[^\n?]+\?)", kg_res, re.M
    )
    questions = list(map(lambda x: x[1], questions))
    return questions


def answer_open_questions(bot: Bot, target: str, sentence: str, prefix):
    kg_prompt = f"""
I am going to ask you about {target}.
{prefix}

Please answer the following question.
{sentence}
    """
    # TODO make more resilient
    with bot as bot_ses:
        bot_ses.set_deterministic(True)
        bot_ses.set_num_answers(1)
        kg_res = bot_ses.ask(kg_prompt)[0]
    return kg_res


def generate_alt_statements_answer_question(
    bot: chatprotect.bots.bot.Bot, target: str, sentence: str, prefix: str
):
    alt_statements = []
    questions = extract_open_questions(bot, target, sentence, prefix)
    # write similar statements next to each other
    for j, question in enumerate(questions):
        # We can just re-use the previous statement!
        # statement = generate_statement(bot, *triple, target, prefix)
        alt_statement = answer_open_questions(bot, target, question, prefix)
        alt_statements.append((f"{question} {alt_statement}", (question)))
    return alt_statements


def generate_alt_statements_prefix_with_triple_wo_object(
    bot: chatprotect.bots.bot.Bot,
    target: str,
    sentence: str,
    prefix: str,
    alts=1,
    override_temperature=None,
    statement_generation_function=generate_statement_missing_object,
):
    triples = [(t, "compactie") for t in extract_triples_compact_ie(sentence)]
    # triples.extend((t, "native") for t in extract_triples(bot, sentence, target))

    alt_statements = []
    # write similar statements next to each other
    for j, (triple, origin) in enumerate(triples):
        # We can just re-use the previous statement!
        # statement = generate_statement(bot, *triple, target, prefix)
        # TODO this assumes chatgpt api
        usage_before = sum(
            u.prompt_tokens + u.completion_tokens for u in bot.total_usage
        )
        alt_statement_res = statement_generation_function(
            bot,
            triple[0],
            triple[1],
            target,
            prefix,
            alts=alts,
            override_temperature=override_temperature,
        )
        usage_after = sum(
            u.prompt_tokens + u.completion_tokens for u in bot.total_usage
        )
        for alt_statement in alt_statement_res:
            alt_statements.append(
                (alt_statement, (triple, origin, (usage_after - usage_before)))
            )
    return alt_statements


def generate_alt_statement_prefix(bot: Bot, target, prefix, n: int):
    alt_statements = generate_statement_prefix(bot, target, prefix, n)
    return [(s, i) for i, s in enumerate(alt_statements)]


def generate_alt_statement_prefix_rephrase(bot: Bot, target, sentence, prefix, n: int):
    alt_statements = generate_statement_prefix_rephrase(
        bot, target, sentence, prefix, n
    )
    return [(s, i) for i, s in enumerate(alt_statements)]
