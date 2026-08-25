"""二分类模型统一评估。"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def evaluate_binary(y_true: pd.Series, score: np.ndarray,
                    lift_fractions: Iterable[float], is_probability: bool = True) -> Tuple[Dict[str, float], pd.DataFrame]:
    """评估概率或纯排序分；纯排序分不计算依赖概率校准的指标。"""
    y = np.asarray(y_true, dtype=float)
    raw_score = np.asarray(score, dtype=float)
    if y.shape[0] != raw_score.shape[0]:
        raise ValueError("标签与预测长度不一致")
    base_rate = float(y.mean())
    metrics = {
        "rows": int(len(y)),
        "positive_rate": base_rate,
        "roc_auc": float(roc_auc_score(y, raw_score)) if np.unique(y).size > 1 else float("nan"),
        "average_precision": float(average_precision_score(y, raw_score)) if y.sum() > 0 else float("nan"),
        "log_loss": float("nan"),
        "brier_score": float("nan"),
    }
    if is_probability:
        probability = np.clip(raw_score, 1e-7, 1 - 1e-7)
        metrics["log_loss"] = float(log_loss(y, probability, labels=[0, 1]))
        metrics["brier_score"] = float(brier_score_loss(y, probability))
    order = np.argsort(-raw_score, kind="mergesort")
    for fraction in lift_fractions:
        k = max(1, int(np.ceil(len(y) * float(fraction))))
        selected = y[order[:k]]
        precision = float(selected.mean())
        metrics[f"precision_at_{fraction:.2f}"] = precision
        metrics[f"recall_at_{fraction:.2f}"] = float(selected.sum() / max(y.sum(), 1))
        metrics[f"lift_at_{fraction:.2f}"] = float(precision / base_rate) if base_rate > 0 else float("nan")

    ranked = pd.DataFrame({"y": y, "score": raw_score}).sort_values(
        "score", ascending=False, kind="mergesort").reset_index(drop=True)
    ranked["decile"] = np.minimum((np.arange(len(ranked)) * 10 // len(ranked)) + 1, 10)
    deciles = ranked.groupby("decile", observed=True).agg(
        samples=("y", "size"), positives=("y", "sum"),
        actual_positive_rate=("y", "mean"), average_score=("score", "mean"),
        min_score=("score", "min"), max_score=("score", "max"),
    ).reset_index()
    return metrics, deciles
