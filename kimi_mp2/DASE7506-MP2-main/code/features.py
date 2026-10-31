import re
from collections import Counter

import numpy as np

from common import parse_answer

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
WORD_RE = re.compile(r"[a-zA-Z]+")
STOP_WORDS = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "on", "for", "how", "many", "much", "what", "does", "do", "he", "she", "they", "his", "her", "their"}


def numbers(text):
    values = []
    for match in NUMBER_RE.finditer(str(text)):
        try:
            value = match.group().replace(",", "")
            if "/" in value:
                numerator, denominator = value.split("/")
                parsed = float(numerator) / float(denominator)
            else:
                parsed = float(value)
            if np.isfinite(parsed):
                values.append(parsed)
        except (ValueError, ZeroDivisionError):
            pass
    return values


def answer_value(answer):
    if answer is None:
        return 0.0
    try:
        if "/" in answer:
            numerator, denominator = answer.split("/")
            return float(numerator) / float(denominator)
        return float(answer)
    except (ValueError, ZeroDivisionError):
        return 0.0


def rank_fraction(values):
    values = np.asarray(values)
    ordered = np.sort(values)
    return np.array([np.flatnonzero(ordered == value).mean() / 3.0 for value in values])


def hand_features(bank, embeddings):
    rows = []
    for row, group_embeddings in zip(bank, embeddings):
        texts = [candidate["text"] for candidate in row["candidates"]]
        answers = [parse_answer(text) for text in texts]
        counts = Counter(answer for answer in answers if answer is not None)
        valid_count = sum(answer is not None for answer in answers)
        question_numbers = numbers(row["question"])
        question_number_set = set(question_numbers)
        question_words = {word.lower() for word in WORD_RE.findall(row["question"])} - STOP_WORDS
        group_numbers = [numbers(text) for text in texts]
        group_words = [{word.lower() for word in WORD_RE.findall(text)} - STOP_WORDS for text in texts]
        lengths = np.array([len(text) for text in texts], dtype=float)
        word_counts = np.array([len(WORD_RE.findall(text)) for text in texts], dtype=float)
        number_counts = np.array([len(items) for items in group_numbers], dtype=float)
        line_counts = np.array([text.count("\n") + 1 for text in texts], dtype=float)
        normalized = group_embeddings / np.maximum(np.linalg.norm(group_embeddings, axis=1, keepdims=True), 1e-8)
        mean_embedding = normalized.mean(axis=0)
        group_cosines = normalized @ mean_embedding / max(np.linalg.norm(mean_embedding), 1e-8)
        cluster_cosines = np.zeros(4)
        cluster_agreement = np.zeros(4)
        for i, answer in enumerate(answers):
            members = [j for j, other in enumerate(answers) if answer is not None and other == answer]
            if members:
                center = normalized[members].mean(axis=0)
                cluster_cosines[i] = normalized[i] @ center / max(np.linalg.norm(center), 1e-8)
                cluster_agreement[i] = len(members) / max(valid_count, 1)
        group_rows = []
        for i, text in enumerate(texts):
            answer = answers[i]
            value = answer_value(answer)
            own_numbers = group_numbers[i]
            own_set = set(own_numbers)
            other_sets = [set(group_numbers[j]) for j in range(4) if j != i]
            jaccards = [len(own_set & other) / max(len(own_set | other), 1) for other in other_sets]
            overlaps = [bool(own_set & other) for other in other_sets]
            final_occurrences = len(NUMBER_RE.findall(text.rsplit("####", 1)[-1]))
            group_rows.append([
                float(answer is not None),
                counts.get(answer, 0) / 4.0,
                cluster_agreement[i],
                len(counts) / 4.0,
                valid_count / 4.0,
                np.log1p(lengths[i]) / 8.0,
                np.log1p(word_counts[i]) / 7.0,
                np.log1p(line_counts[i]) / 5.0,
                np.log1p(number_counts[i]) / 5.0,
                sum(character.isdigit() for character in text) / max(len(text), 1),
                min(text.count("="), 20) / 20.0,
                min(text.count("\\"), 40) / 40.0,
                float("\\boxed" in text),
                float("####" in text),
                float("answer" in text.lower()),
                float("therefore" in text.lower() or "thus" in text.lower()),
                len(question_number_set & own_set) / max(len(question_number_set), 1),
                len(question_number_set & own_set) / max(len(own_set), 1),
                len(question_words & group_words[i]) / max(len(question_words), 1),
                len(group_words[i]) / max(len(WORD_RE.findall(text)), 1),
                np.sign(value),
                np.log1p(abs(value)) / 12.0,
                float(value.is_integer()),
                rank_fraction(lengths)[i],
                rank_fraction(word_counts)[i],
                rank_fraction(number_counts)[i],
                rank_fraction(line_counts)[i],
                group_cosines[i],
                cluster_cosines[i],
                np.mean(jaccards),
                np.max(jaccards),
                np.mean(overlaps),
                min(final_occurrences, 10) / 10.0,
            ])
        rows.append(group_rows)
    return np.round(np.asarray(rows, dtype=np.float32), decimals=6)


def make_features(bank, embeddings, mode="base"):
    from math_features import equation_features

    embeddings = embeddings.astype(np.float32)
    hand = hand_features(bank, embeddings)
    parts = [embeddings]
    if mode == "centered":
        parts.append(embeddings - embeddings.mean(axis=1, keepdims=True))
    elif mode not in ("base", "nomath"):
        raise ValueError(f"Unknown feature mode: {mode}")
    parts.append(hand)
    if mode != "nomath":
        parts.append(equation_features(bank))
    return np.concatenate(parts, axis=2)


def load_feature_split(split, loader, mode="base"):
    bank, arrays = loader(split)
    return bank, make_features(bank, arrays["embeddings"], mode)
