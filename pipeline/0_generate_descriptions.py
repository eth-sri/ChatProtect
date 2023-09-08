import concurrent.futures
import json
from argparse import ArgumentParser
from dataclasses import asdict

from chatprotect.util import Description, fetch_model, prompt_identifier
import pathlib

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--glm-model",
        type=str,
        help="select the generating model",
        default="chatgpt",
    )
    ap.add_argument(
        "--amount",
        type=int,
        help="amount of descriptions to generate",
        default=3,
    )
    ap.add_argument(
        "--prompt",
        type=str,
        help="Prompt to generate answers for",
        default="Please tell me about Thomas Chapais",
    )
    ap.add_argument(
        "--test_description_dir",
        default="test/custom/descriptions",
        help="place to store descriptions",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing descriptions",
    )
    args = ap.parse_args()
    model = args.glm_model
    description_dir = pathlib.Path(args.test_description_dir)
    bot = fetch_model(model)

    def generate_topic(topic):
        topic_lower = prompt_identifier(topic)
        target_dir = description_dir / model / topic_lower
        if target_dir.is_dir() and not args.force:
            # skip descriptions that already exist
            print(
                f"Skipping {topic} as descriptions are already present and --force not set"
            )
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.amount):
            bot.seed = i
            with bot as bot_ses:
                bot_ses.set_deterministic(False)
                bot_ses.set_num_answers(1)
                if "###" in topic:
                    user, assistant = [x.strip() for x in topic.split("###")]
                    bot_ses.add_history(user, assistant)
                    desc = bot_ses.ask("")[0]
                else:
                    desc = bot_ses.ask(topic)[0]

            d = Description(
                prompt=topic,
                cost_model=bot_ses.usage.completion_tokens,
                cost_user=bot_ses.usage.prompt_tokens,
                text=desc,
            )
            with (target_dir / (str(i) + ".txt")).open("w") as f:
                json.dump(asdict(d), f, indent=2)

    if args.prompt:
        topics = [args.prompt]
    print(f"Writing results to {description_dir} for {len(topics)} entities")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_topic, topic) for topic in topics]
