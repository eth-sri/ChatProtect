import copy

from evaluate import load

from argparse import ArgumentParser
import pathlib

from chatprotect.util import read_desc_file, read_entities

parser = ArgumentParser()
parser.add_argument(
    "--test_description_dir",
    type=str,
    help="Input dir",
    default="./test/custom/descriptions",
)
parser.add_argument(
    "--mode",
    type=int,
    help="revision mode",
    default=-1,
)
args = parser.parse_args()
input_texts = []
entity_lens = []
entities = []
test_description_dir = pathlib.Path(args.test_description_dir)
for entity_dir in test_description_dir.iterdir():
    entities.append(entity_dir.stem)
    entity_texts = []
    if args.mode >= 0:
        entity_dir /= f"m{args.mode}"
    for desc_file in entity_dir.iterdir():
        try:
            desc = read_desc_file(None, desc_file)
        except ValueError:
            continue
        if desc.text:
            entity_texts.append(desc)
    input_texts.extend(entity_texts)
    entity_lens.append(len(entity_texts))
input_texts = [i.text for i in input_texts]

perplexity = load("perplexity", module_type="measurement")
results = perplexity.compute(
    data=input_texts, model_id="facebook/opt-1.3b", batch_size=5
)
all_ppls = results["perplexities"]
ppls = copy.copy(all_ppls)
for l, e in zip(entity_lens, entities):
    res = ppls[:l]
    print(
        {
            "perplexities": res,
            "mean_perplexity": (sum(res) / len(res)) if l > 0 else 0,
            "entity": e,
        }
    )
    ppls = ppls[l:]
print(sum(all_ppls) / len(all_ppls))
