from __future__ import annotations


def classification_report(pairs: list[tuple[str | None, str]], labels: list[str]) -> dict:
    """Accuracy, per-class P/R/F1 and confusion matrix from (pred, gold) pairs."""
    n = len(pairs)
    seen = sorted({g for _, g in pairs} | {p for p, _ in pairs if p}, key=lambda x: (x not in labels, labels.index(x) if x in labels else 0, x))
    confusion = {g: {p: 0 for p in seen} for g in seen}
    for p, g in pairs:
        if p:
            confusion[g][p] += 1
    per_class = {}
    for lab in seen:
        tp = confusion[lab][lab]
        fp = sum(confusion[g][lab] for g in seen) - tp
        fn = sum(confusion[lab].values()) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": tp + fn}
    supported = [v for v in per_class.values() if v["support"]]
    correct = sum(p == g for p, g in pairs)
    return {
        "n": n,
        "accuracy": round(correct / n, 3) if n else None,
        "macro_f1": round(sum(v["f1"] for v in supported) / len(supported), 3) if supported else None,
        "labels": seen,
        "per_class": per_class,
        "confusion": confusion,
    }
