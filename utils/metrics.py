import numpy as np
from collections import Counter


def compute_bleu(predictions, references, max_n=4):
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if len(pred_tokens) == 0:
            scores.append(0.0)
            continue
        clipped_counts = 0
        total_counts = 0
        for n in range(1, max_n + 1):
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)
            for ngram, count in pred_ngrams.items():
                clipped_counts += min(count, ref_ngrams.get(ngram, 0))
                total_counts += count
        if total_counts == 0:
            scores.append(0.0)
        else:
            precision = clipped_counts / total_counts
            bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
            scores.append(bp * precision)
    return np.mean(scores) * 100 if scores else 0.0


def _get_ngrams(tokens, n):
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return ngrams


def compute_rouge(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        if len(ref_tokens) == 0:
            scores.append(0.0)
            continue
        lcs_len = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs_len / max(len(pred_tokens), 1)
        recall = lcs_len / max(len(ref_tokens), 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)
    return np.mean(scores) * 100 if scores else 0.0


def _lcs_length(x, y):
    m, n = len(x), len(y)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[m][n]


def compute_meteor(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if len(ref_tokens) == 0:
            scores.append(0.0)
            continue
        matches = pred_tokens & ref_tokens
        if len(matches) == 0:
            scores.append(0.0)
            continue
        precision = len(matches) / max(len(pred_tokens), 1)
        recall = len(matches) / max(len(ref_tokens), 1)
        f1 = (10 * precision * recall) / (9 * precision + recall + 1e-8)
        chunks = _count_chunks(pred.lower().split(), ref.lower().split(), matches)
        penalty = 0.5 * (chunks / max(len(matches), 1)) ** 3
        score = f1 * (1 - penalty)
        scores.append(max(score, 0.0))
    return np.mean(scores) * 100 if scores else 0.0


def _count_chunks(pred_tokens, ref_tokens, matches):
    chunks = 0
    in_chunk = False
    ref_set = set(ref_tokens)
    for token in pred_tokens:
        if token in matches:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False
    return chunks


def compute_accuracy(predictions, references):
    if not predictions:
        return 0.0
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_clean = pred.strip().lower()
        ref_clean = ref.strip().lower()
        if pred_clean == ref_clean:
            correct += 1
        elif ref_clean in pred_clean or pred_clean in ref_clean:
            correct += 1
    return correct / len(predictions) * 100
