"""双轨 RFM 特征与基线预测。

所有阈值和响应率只允许在训练集拟合，验证集和测试集仅做 transform。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PLATFORM_SEGMENT_ORDER = [
    "Champions", "Loyal Accounts", "Potential Loyalist", "Promising",
    "New Active Accounts", "Need Attention", "About to Sleep", "At Risk",
    "Low Spenders", "Lost", "Unclassified",
]


def _strict_quantile_edges(series: pd.Series, n_bins: int) -> List[float]:
    """返回严格递增的训练集分位点；相同原值不会被拆到不同档。"""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return []
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    raw = np.quantile(values, quantiles)
    return [float(x) for x in np.unique(raw)]


def _score_by_edges(series: pd.Series, edges: Iterable[float], reverse: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    edges = np.asarray(list(edges), dtype=float)
    score = pd.Series(np.searchsorted(edges, values.to_numpy(dtype=float), side="left") + 1,
                      index=series.index, dtype="Int64")
    if reverse:
        score = len(edges) + 2 - score
    return score.mask(values.isna())


def _classify_platform(r: int, f: int, m: int) -> str:
    if r in (4, 5) and f in (4, 5) and m in (4, 5):
        return "Champions"
    if r in (4, 5) and f in (1, 2) and m in (3, 4, 5):
        return "Promising"
    if r in (3, 4, 5) and f in (3, 4, 5) and m in (3, 4, 5):
        return "Loyal Accounts"
    if r in (3, 4, 5) and f in (2, 3) and m in (2, 3, 4):
        return "Potential Loyalist"
    if r == 5 and f == 1:
        return "New Active Accounts"
    if r in (3, 4, 5) and m in (1, 2):
        return "Low Spenders"
    if r in (2, 3) and f in (1, 2) and m in (4, 5):
        return "Need Attention"
    if r in (2, 3) and f in (1, 2) and m in (1, 2, 3):
        return "About to Sleep"
    if r in (1, 2) and m in (3, 4, 5):
        return "At Risk"
    if r in (1, 2) and m in (1, 2):
        return "Lost"
    return "Unclassified"


@dataclass
class PlatformRFM:
    mature_min_total_orders: int = 3
    n_bins: int = 5
    edges_: Dict[str, List[float]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "PlatformRFM":
        mature = frame[pd.to_numeric(frame["total_orders"], errors="coerce") >= self.mature_min_total_orders]
        self.edges_ = {
            "r": _strict_quantile_edges(mature["l720d_last_to_today_days"], self.n_bins),
            "f": _strict_quantile_edges(mature["l360d_buy_days"], self.n_bins),
            "m": _strict_quantile_edges(mature["l360d_dgmv"], self.n_bins),
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.edges_:
            raise RuntimeError("PlatformRFM 尚未 fit")
        result = pd.DataFrame(index=frame.index)
        total_orders = pd.to_numeric(frame["total_orders"], errors="coerce")
        result["platform_lifecycle"] = np.select(
            [total_orders.isna(), total_orders.eq(0), total_orders.between(1, self.mature_min_total_orders - 1),
             total_orders.ge(self.mature_min_total_orders)],
            ["U", "N", "NEW", "MATURE"], default="U",
        )
        mature = result["platform_lifecycle"].eq("MATURE")
        result["platform_r_score"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        result["platform_f_score"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        result["platform_m_score"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        result.loc[mature, "platform_r_score"] = _score_by_edges(
            frame.loc[mature, "l720d_last_to_today_days"], self.edges_["r"], reverse=True)
        result.loc[mature, "platform_f_score"] = _score_by_edges(
            frame.loc[mature, "l360d_buy_days"], self.edges_["f"])
        result.loc[mature, "platform_m_score"] = _score_by_edges(
            frame.loc[mature, "l360d_dgmv"], self.edges_["m"])
        valid = mature & result[["platform_r_score", "platform_f_score", "platform_m_score"]].notna().all(axis=1)
        result["platform_rfm_sum"] = pd.Series(np.nan, index=frame.index)
        result.loc[valid, "platform_rfm_sum"] = result.loc[valid, [
            "platform_r_score", "platform_f_score", "platform_m_score"]].astype(float).sum(axis=1)
        result["platform_rfm_cell"] = result["platform_lifecycle"]
        result.loc[valid, "platform_rfm_cell"] = (
            "M_R" + result.loc[valid, "platform_r_score"].astype(str)
            + "F" + result.loc[valid, "platform_f_score"].astype(str)
            + "M" + result.loc[valid, "platform_m_score"].astype(str)
        )
        result["platform_segment"] = result["platform_lifecycle"].map({
            "N": "No Purchase", "NEW": "New/Cultivation", "U": "Unknown", "MATURE": "Unclassified"})
        result.loc[valid, "platform_segment"] = [
            _classify_platform(int(r), int(f), int(m))
            for r, f, m in result.loc[valid, ["platform_r_score", "platform_f_score", "platform_m_score"]].itertuples(index=False)
        ]
        return result


@dataclass
class LiveRFM:
    r_bins: List[float] = field(default_factory=lambda: [7, 30, 180])
    n_bins: int = 3
    f_edges_: List[float] = field(default_factory=list)
    m_edges_: List[float] = field(default_factory=list)
    probability_maps_: Dict[int, Dict[str, float]] = field(default_factory=dict)
    lifecycle_maps_: Dict[int, Dict[str, float]] = field(default_factory=dict)
    global_rate_: Optional[float] = None
    selected_alpha_: Optional[int] = None

    def fit_scores(self, frame: pd.DataFrame) -> "LiveRFM":
        f = pd.to_numeric(frame["l180d_live_order_cnt"], errors="coerce")
        m = pd.to_numeric(frame["l180d_live_dgmv"], errors="coerce")
        self.f_edges_ = _strict_quantile_edges(f[f > 0], self.n_bins)
        self.m_edges_ = _strict_quantile_edges(m[m > 0], self.n_bins)
        return self

    def transform_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.f_edges_ or not self.m_edges_:
            raise RuntimeError("LiveRFM 尚未 fit_scores")
        result = pd.DataFrame(index=frame.index)
        r = pd.to_numeric(frame["last_goods_live_buy_to_today_days"], errors="coerce")
        f = pd.to_numeric(frame["l180d_live_order_cnt"], errors="coerce")
        m = pd.to_numeric(frame["l180d_live_dgmv"], errors="coerce")
        stage = frame["customer_stage_live_yesterday"].fillna("unknown").astype(str)
        invalid = r.isna() | f.isna() | m.isna() | (f < 0) | (m < 0)
        active = (~invalid) & (f > 0) & r.between(0, self.r_bins[-1])
        dormant = (~invalid) & f.eq(0) & stage.eq("直播180天流失")
        no_history = (~invalid) & f.eq(0) & stage.eq("直播潜客")
        result["live_lifecycle"] = np.select([active, dormant, no_history], ["A", "D", "N"], default="U")
        result["live_r_score"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        result["live_f_score"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        result["live_m_score"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        result.loc[active, "live_r_score"] = pd.cut(
            r[active], bins=[-np.inf, self.r_bins[0], self.r_bins[1], self.r_bins[2]],
            labels=[3, 2, 1], include_lowest=True).astype("Int64")
        result.loc[active, "live_f_score"] = _score_by_edges(f[active], self.f_edges_)
        result.loc[active, "live_m_score"] = _score_by_edges(m[active], self.m_edges_)
        zero_gmv = active & m.eq(0)
        result.loc[zero_gmv, "live_m_score"] = 1
        result["zero_gmv_with_order"] = zero_gmv.astype("int8")
        result["live_rfm_sum"] = pd.Series(np.nan, index=frame.index)
        result.loc[active, "live_rfm_sum"] = result.loc[active, [
            "live_r_score", "live_f_score", "live_m_score"]].astype(float).sum(axis=1)
        result["live_rfm_cell"] = result["live_lifecycle"]
        result.loc[active, "live_rfm_cell"] = (
            "A_R" + result.loc[active, "live_r_score"].astype(str)
            + "F" + result.loc[active, "live_f_score"].astype(str)
            + "M" + result.loc[active, "live_m_score"].astype(str)
        )
        return result

    def fit_probability(self, scored_train: pd.DataFrame, target: pd.Series,
                        alphas: Iterable[int]) -> "LiveRFM":
        y = pd.to_numeric(target, errors="coerce")
        if y.isna().any():
            raise ValueError("训练标签存在空值或非数值")
        self.global_rate_ = float(y.mean())
        stats = pd.DataFrame({"cell": scored_train["live_rfm_cell"], "lifecycle": scored_train["live_lifecycle"], "y": y})
        cell_stats = stats.groupby("cell", observed=True)["y"].agg(["sum", "count"])
        life_stats = stats.groupby("lifecycle", observed=True)["y"].agg(["sum", "count"])
        for alpha in alphas:
            self.probability_maps_[int(alpha)] = (
                (cell_stats["sum"] + alpha * self.global_rate_) / (cell_stats["count"] + alpha)).to_dict()
            self.lifecycle_maps_[int(alpha)] = (
                (life_stats["sum"] + alpha * self.global_rate_) / (life_stats["count"] + alpha)).to_dict()
        return self

    def predict_probability(self, scored: pd.DataFrame, alpha: int) -> pd.Series:
        if self.global_rate_ is None or int(alpha) not in self.probability_maps_:
            raise RuntimeError("LiveRFM 尚未 fit_probability，或 alpha 未拟合")
        cell = scored["live_rfm_cell"].map(self.probability_maps_[int(alpha)])
        lifecycle = scored["live_lifecycle"].map(self.lifecycle_maps_[int(alpha)])
        return cell.fillna(lifecycle).fillna(self.global_rate_).astype(float)
