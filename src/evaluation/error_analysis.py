"""
Error analysis utilities for NER.

Helps identify common error patterns:
- Boundary errors (partial entity matches)
- Type confusion (correct span, wrong entity type)
- Missing entities (false negatives)
- Spurious entities (false positives)
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _extract_entities(labels: List[str]) -> List[Tuple[str, int, int]]:
    """
    Extract entities as (type, start, end) tuples from BIO label sequence.
    """
    entities = []
    current_type = None
    start = None

    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current_type is not None:
                entities.append((current_type, start, i))
            current_type = label[2:]
            start = i
        elif label.startswith("I-"):
            if current_type is None or label[2:] != current_type:
                # Malformed: I- without matching B-; treat as new entity
                if current_type is not None:
                    entities.append((current_type, start, i))
                current_type = label[2:]
                start = i
        else:  # "O"
            if current_type is not None:
                entities.append((current_type, start, i))
                current_type = None
                start = None

    if current_type is not None:
        entities.append((current_type, start, len(labels)))

    return entities


def analyse_errors(
    tokens_list: List[List[str]],
    true_labels_list: List[List[str]],
    pred_labels_list: List[List[str]],
) -> Dict:
    """
    Perform error analysis on NER predictions.

    Returns
    -------
    dict with keys:
        - boundary_errors: entities with correct type but wrong boundaries
        - type_errors: entities with correct span but wrong type
        - false_negatives: gold entities not predicted
        - false_positives: predicted entities not in gold
        - per_type_stats: per-entity-type TP/FP/FN counts
    """
    boundary_errors = []
    type_errors = []
    false_negatives = []
    false_positives = []
    per_type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for tokens, true_labels, pred_labels in zip(tokens_list, true_labels_list, pred_labels_list):
        true_entities = set(_extract_entities(true_labels))
        pred_entities = set(_extract_entities(pred_labels))

        # True positives
        tp = true_entities & pred_entities
        for etype, start, end in tp:
            per_type_stats[etype]["tp"] += 1

        # False negatives
        fn = true_entities - pred_entities
        for etype, start, end in fn:
            per_type_stats[etype]["fn"] += 1
            span_text = " ".join(tokens[start:end])
            # Check if it's a boundary or type error
            matched = False
            for ptype, pstart, pend in pred_entities:
                if start == pstart and end == pend and ptype != etype:
                    type_errors.append({
                        "tokens": span_text,
                        "true_type": etype,
                        "pred_type": ptype,
                    })
                    matched = True
                elif ptype == etype and (
                    (pstart <= start < pend) or (pstart < end <= pend)
                ):
                    boundary_errors.append({
                        "tokens": span_text,
                        "true_span": (start, end),
                        "pred_span": (pstart, pend),
                        "type": etype,
                    })
                    matched = True
            if not matched:
                false_negatives.append({
                    "tokens": span_text,
                    "type": etype,
                    "span": (start, end),
                })

        # False positives
        fp = pred_entities - true_entities
        for ptype, pstart, pend in fp:
            per_type_stats[ptype]["fp"] += 1
            span_text = " ".join(tokens[pstart:pend])
            # Only count as FP if not already counted as boundary/type error
            if not any(
                (pstart <= start < pend) or (pstart < end <= pend)
                for etype, start, end in true_entities
            ):
                false_positives.append({
                    "tokens": span_text,
                    "type": ptype,
                    "span": (pstart, pend),
                })

    return {
        "boundary_errors": boundary_errors,
        "type_errors": type_errors,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "per_type_stats": dict(per_type_stats),
    }


def print_error_report(analysis: Dict, top_k: int = 20) -> str:
    """Format error analysis results into a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("NER ERROR ANALYSIS REPORT")
    lines.append("=" * 60)

    # Per-type summary
    lines.append("\nPer-Entity-Type Performance:")
    lines.append(f"{'Type':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    lines.append("-" * 60)
    for etype, stats in sorted(analysis["per_type_stats"].items()):
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        lines.append(f"{etype:<20} {tp:>6} {fp:>6} {fn:>6} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

    # Error summaries
    lines.append(f"\nBoundary Errors: {len(analysis['boundary_errors'])}")
    lines.append(f"Type Errors: {len(analysis['type_errors'])}")
    lines.append(f"False Negatives (missed): {len(analysis['false_negatives'])}")
    lines.append(f"False Positives (spurious): {len(analysis['false_positives'])}")

    # Top false negatives
    if analysis["false_negatives"]:
        lines.append(f"\nTop {top_k} Missed Entities (False Negatives):")
        fn_counter = Counter(e["tokens"] for e in analysis["false_negatives"])
        for text, count in fn_counter.most_common(top_k):
            lines.append(f"  {text} (x{count})")

    # Top false positives
    if analysis["false_positives"]:
        lines.append(f"\nTop {top_k} Spurious Entities (False Positives):")
        fp_counter = Counter(e["tokens"] for e in analysis["false_positives"])
        for text, count in fp_counter.most_common(top_k):
            lines.append(f"  {text} (x{count})")

    report = "\n".join(lines)
    return report
