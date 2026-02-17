"""
Entity-level evaluation metrics for NER.

Uses seqeval for strict entity-level precision, recall, and F1,
which is the standard evaluation for biomedical NER benchmarks.

Falls back to a built-in implementation if seqeval is not installed.
"""

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    from seqeval.scheme import IOB2

    _HAS_SEQEVAL = True
except ImportError:
    _HAS_SEQEVAL = False
    logger.warning("seqeval not installed; using built-in entity-level metrics.")


# ---------------------------------------------------------------------------
# Built-in entity extraction and metric computation (seqeval fallback)
# ---------------------------------------------------------------------------

def _extract_entities_from_bio(labels: List[str]) -> set:
    """Extract (entity_type, start, end) tuples from a BIO label sequence."""
    entities = set()
    current_type = None
    start = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current_type is not None:
                entities.add((current_type, start, i))
            current_type = label[2:]
            start = i
        elif label.startswith("I-"):
            if current_type is None or label[2:] != current_type:
                if current_type is not None:
                    entities.add((current_type, start, i))
                current_type = label[2:]
                start = i
        else:
            if current_type is not None:
                entities.add((current_type, start, i))
                current_type = None
    if current_type is not None:
        entities.add((current_type, start, len(labels)))
    return entities


def _compute_f1_from_lists(
    true_labels: List[List[str]], pred_labels: List[List[str]],
) -> Dict[str, float]:
    """Compute entity-level precision, recall, F1 from label lists."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_ents = _extract_entities_from_bio(true_seq)
        pred_ents = _extract_entities_from_bio(pred_seq)
        tp = true_ents & pred_ents
        fp = pred_ents - true_ents
        fn = true_ents - pred_ents
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)
        for etype, _, _ in tp:
            per_type[etype]["tp"] += 1
        for etype, _, _ in fp:
            per_type[etype]["fp"] += 1
        for etype, _, _ in fn:
            per_type[etype]["fn"] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Build a classification report string
    lines = [f"{'Type':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}"]
    lines.append("-" * 52)
    for etype in sorted(per_type):
        s = per_type[etype]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        lines.append(f"{etype:<20} {p:>8.4f} {r:>8.4f} {f:>8.4f} {tp + fn:>8}")
    lines.append("-" * 52)
    lines.append(f"{'micro avg':<20} {precision:>8.4f} {recall:>8.4f} {f1:>8.4f} {total_tp + total_fn:>8}")
    report = "\n".join(lines)

    return {"precision": precision, "recall": recall, "f1": f1, "report": report}


def compute_ner_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_list: List[str],
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute entity-level NER metrics.

    Parameters
    ----------
    predictions : np.ndarray
        (num_examples, seq_len) predicted label indices.
    labels : np.ndarray
        (num_examples, seq_len) true label indices, with -100 for ignored.
    label_list : list of str
        Ordered label strings.
    ignore_index : int
        Label id to ignore (special/subword tokens).

    Returns
    -------
    dict
        Keys: precision, recall, f1, report (str).
    """
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_filtered = []
        for p, l in zip(pred_seq, label_seq):
            if l == ignore_index:
                continue
            true_seq.append(label_list[l])
            pred_seq_filtered.append(label_list[p] if p < len(label_list) else "O")
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_filtered)

    if _HAS_SEQEVAL:
        precision = precision_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
        recall = recall_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
        f1 = f1_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
        report = classification_report(true_labels, pred_labels, mode="strict", scheme=IOB2)
        return {"precision": precision, "recall": recall, "f1": f1, "report": report}
    else:
        return _compute_f1_from_lists(true_labels, pred_labels)


def build_compute_metrics_fn(label_list: List[str]) -> Callable:
    """
    Build a compute_metrics function compatible with HuggingFace Trainer.

    The returned function takes an EvalPrediction and returns a dict of metrics.
    """

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metrics = compute_ner_metrics(predictions, labels, label_list)
        # Remove the full report from metrics dict (too verbose for logging)
        report = metrics.pop("report", "")
        logger.info("\n%s", report)
        return metrics

    return compute_metrics
