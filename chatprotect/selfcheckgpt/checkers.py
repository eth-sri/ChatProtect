import abc
import en_core_web_sm
import torch
import bert_score
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import (
    LongformerTokenizer,
    LongformerForMultipleChoice,
    LongformerForSequenceClassification,
)
import logging

from chatprotect.util import CHECKED, INCONSISTENT
from .modeling_mqag import MQAGConfig, question_generation_sentence_level, answering

SELF_CHECKER = None
_LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def train(args, sents):
    global SELF_CHECKER
    if SELF_CHECKER is None:
        if args.mode in (40, 42):
            SELF_CHECKER = SelfCheckUnigram(True)
        elif args.mode in (41, 43):
            SELF_CHECKER = SelfCheckUnigram(False)
        else:
            assert False

    for sent in sents:
        SELF_CHECKER.add(sent)
    SELF_CHECKER.train()


@torch.no_grad()
def check(args, stmt1, stmt2, target, prefix):
    global SELF_CHECKER
    if SELF_CHECKER is None:
        if args.mode == 5:
            SELF_CHECKER = SelfCheckBERTScore()
        elif args.mode == 6:
            SELF_CHECKER = SelfCheckMQAG()
        else:
            assert False

    stmt1 = stmt1.strip()
    stmt2 = stmt2.strip()
    prefix = prefix.strip()
    return SELF_CHECKER.check(stmt1, stmt2, target, prefix)


@torch.no_grad()
def check_multi(args, stmt1, stmt2s, target, prefix):
    global SELF_CHECKER
    if SELF_CHECKER is None:
        if args.mode == 12:
            SELF_CHECKER = SelfCheckBERTScore()
        elif args.mode == 13:
            SELF_CHECKER = SelfCheckMQAG()
        else:
            assert False

    stmt1 = stmt1.strip()
    stmt2 = [s.strip() for s in stmt2s]
    prefix = prefix.strip()
    return SELF_CHECKER.check_multi(stmt1, stmt2, target, prefix)


class SelfCheckBERTScore:
    def __init__(self):
        self.scorer = bert_score.BERTScorer(lang="en", nthreads=1)
        _LOGGER.info("SelfCheck-BERTScore initialized")

    def check(self, stmt1, stmt2, target, prefix):
        p, r, f1 = self.scorer.score([stmt2], [stmt1])
        prob = 1 - f1.item()

        return prob

    def check_multi(self, stmt1, stmt2s, target, prefix):
        s = 0
        for stmt2 in stmt2s:
            s += 1 - self.check(stmt1, stmt2, target, prefix)

        return 1 - s / len(stmt2s)


def answerability_scoring(
    u_model,
    u_tokenizer,
    question,
    context,
    max_length,
    device,
):
    """
    :return prob: prob -> 0.0 means unanswerable, prob -> 1.0 means answerable
    """
    input_text = question + " " + u_tokenizer.sep_token + " " + context
    inputs = u_tokenizer(
        input_text, max_length=max_length, truncation=True, return_tensors="pt"
    )
    inputs = inputs.to(device)
    logits = u_model(**inputs).logits
    logits = logits.squeeze(-1)
    prob = torch.sigmoid(logits).item()
    return prob


class SelfCheckMQAG:
    NUM_QUESTIONS = 10
    MAX_SEQ_LENGTH = 4096
    BETA1 = 0.8
    BETA2 = 0.8

    def __init__(self):
        g1_model = MQAGConfig.generation1_squad
        g2_model = MQAGConfig.generation2
        answering_model = MQAGConfig.answering
        answerability_model = MQAGConfig.answerability

        # Question Generation Systems (G1 & G2)
        self.g1_tokenizer = AutoTokenizer.from_pretrained(g1_model)
        self.g1_model = AutoModelForSeq2SeqLM.from_pretrained(g1_model)
        self.g2_tokenizer = AutoTokenizer.from_pretrained(g2_model)
        self.g2_model = AutoModelForSeq2SeqLM.from_pretrained(g2_model)

        # Question Answering System (A)
        self.a_tokenizer = LongformerTokenizer.from_pretrained(answering_model)
        self.a_model = LongformerForMultipleChoice.from_pretrained(answering_model)

        # (Un)Answerability System (U)
        self.u_tokenizer = LongformerTokenizer.from_pretrained(answerability_model)
        self.u_model = LongformerForSequenceClassification.from_pretrained(
            answerability_model
        )

        self.g1_model.eval()
        self.g2_model.eval()
        self.a_model.eval()
        self.u_model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.g1_model.to(device)
        self.g2_model.to(device)
        self.a_model.to(device)
        self.u_model.to(device)
        self.device = device
        _LOGGER.info(f"SelfCheck-MQAG initialized to device {device}")

    def check(self, stmt1, stmt2, target, prefix):
        passage = prefix + " " + stmt1
        sampled_passage = prefix + " " + stmt2

        questions = question_generation_sentence_level(
            self.g1_model,
            self.g1_tokenizer,
            self.g2_model,
            self.g2_tokenizer,
            stmt1,
            passage,
            self.NUM_QUESTIONS,
            self.device,
        )

        count_match, count_mismatch = 0, 0
        for question_item in questions:
            question, options = question_item["question"], question_item["options"]
            # response
            prob = answering(
                self.a_model,
                self.a_tokenizer,
                question,
                options,
                passage,
                self.MAX_SEQ_LENGTH,
                self.device,
            )

            u_score = answerability_scoring(
                self.u_model,
                self.u_tokenizer,
                question,
                passage,
                self.MAX_SEQ_LENGTH,
                self.device,
            )

            prob_s = answering(
                self.a_model,
                self.a_tokenizer,
                question,
                options,
                sampled_passage,
                self.MAX_SEQ_LENGTH,
                self.device,
            )
            u_score_s = answerability_scoring(
                self.u_model,
                self.u_tokenizer,
                question,
                sampled_passage,
                self.MAX_SEQ_LENGTH,
                self.device,
            )

            a = np.argmax(prob)
            a_s = np.argmax(prob_s)
            if a == a_s:
                count_match += u_score_s
            else:
                count_mismatch += u_score_s

        gamma1 = self.BETA2 / (1.0 - self.BETA1)
        gamma2 = self.BETA1 / (1.0 - self.BETA2)
        score = (gamma2**count_mismatch) / (
            (gamma1**count_match) + (gamma2**count_mismatch)
        )

        return score

    def check_multi(self, stmt1, stmt2s, target, prefix):
        passage = prefix + " " + stmt1

        questions = question_generation_sentence_level(
            self.g1_model,
            self.g1_tokenizer,
            self.g2_model,
            self.g2_tokenizer,
            stmt1,
            passage,
            self.NUM_QUESTIONS,
            self.device,
        )

        scores = []
        for question_item in questions:
            question, options = question_item["question"], question_item["options"]

            # response
            prob = answering(
                self.a_model,
                self.a_tokenizer,
                question,
                options,
                passage,
                self.MAX_SEQ_LENGTH,
                self.device,
            )
            u_score = answerability_scoring(
                self.u_model,
                self.u_tokenizer,
                question,
                passage,
                self.MAX_SEQ_LENGTH,
                self.device,
            )
            a = np.argmax(prob)

            count_match, count_mismatch = 0, 0
            for stmt2 in stmt2s:
                sampled_passage = prefix + " " + stmt2
                prob_s = answering(
                    self.a_model,
                    self.a_tokenizer,
                    question,
                    options,
                    sampled_passage,
                    self.MAX_SEQ_LENGTH,
                    self.device,
                )
                u_score_s = answerability_scoring(
                    self.u_model,
                    self.u_tokenizer,
                    question,
                    sampled_passage,
                    self.MAX_SEQ_LENGTH,
                    self.device,
                )
                a_s = np.argmax(prob_s)

                if a == a_s:
                    count_match += u_score_s
                else:
                    count_mismatch += u_score_s

            gamma1 = self.BETA2 / (1.0 - self.BETA1)
            gamma2 = self.BETA1 / (1.0 - self.BETA2)
            score = (gamma2**count_mismatch) / (
                (gamma1**count_match) + (gamma2**count_mismatch)
            )
            scores.append(score)

        return sum(scores) / len(scores)


class SelfCheckUnigram:
    def __init__(self, is_avg):
        self.nlp = en_core_web_sm.load()
        self.token_count = 0
        self.counts = {"<unk>": 0}
        self.is_avg = is_avg

    def add(self, sent):
        tokens = [token.text.lower() for token in self.nlp(sent)]
        self.token_count += len(tokens)
        for token in tokens:
            if token not in self.counts:
                self.counts[token] = 1
            else:
                self.counts[token] += 1

    def train(self):
        self.probs = dict()
        for token, count in self.counts.items():
            self.probs[token] = count / self.token_count

    def check(self, stmt1, stmt2, target, prefix):
        tokens = [token.text.lower() for token in self.nlp(stmt1)]
        logprobs = []
        for token in tokens:
            if token not in self.probs:
                token = "<unk>"
            train_prob = self.probs[token]
            logprob = np.log(train_prob)
            logprobs.append(logprob)
        res = -1.0 * np.mean(logprobs) if self.is_avg else -1.0 * np.min(logprobs)
        return res

    def check_multi(self, stmt1, stmt2s, target, prefix):
        return self.check(stmt1, None, target, prefix)
