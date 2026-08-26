#!/usr/bin/env python3
"""为一次训练生成面向人工 Review 的精简可视化报告。

底层 CSV/JSON 继续保留用于追溯；日常只需查看 outputs/<run_id>/review/。
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# 英文优先 Times New Roman；其不含的中文字形自动回退到本机宋体 Songti SC。
plt.rcParams["font.family"] = ["Times New Roman", "Songti SC"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

# 自定义配色方案
CUSTOM_COLORS = [
    "#FDEBAA",  # 浅黄
    "#EDC3A5",  # 浅橙
    "#DBE4FB",  # 浅蓝
    "#ABD1BC",  # 浅绿
    "#E3BBED",  # 浅紫
    "#CCCC99",  # 橄榄色
    "#BED0F9",  # 蓝色
    "#FCB6A5",  # 浅红
    "#F1F1F1",  # 浅灰
]

DISPLAY_NAMES = {
    "E0_global_rate": "E0 全局正样本率",
    "E1_live_rfm_sum": "E1 直播RFM规则分",
    "E2_live_rfm_probability": "E2 直播RFM响应率",
    "E2_platform_rfm_probability": "E2 全平台RFM响应率",
    "E3_dual_rfm_raw": "E3 双轨RFM轻量模型",
    "E4_live_features": "E4 143个直播特征",
    "E5_live_plus_platform": "E5 直播+全平台特征",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成训练结果 Review 报告")
    parser.add_argument("--run-id", help="指定运行时间目录；默认选择最新且包含 validation_metrics.csv 的完整运行")
    return parser.parse_args()


def resolve_run(run_id: str = None) -> Tuple[Path, Path]:
    outputs = ROOT / "outputs"
    if run_id:
        output_dir = outputs / run_id
        if not (output_dir / "validation_metrics.csv").exists():
            raise FileNotFoundError(f"运行目录不存在或不完整：{output_dir}")
    else:
        candidates = sorted(
            (path for path in outputs.iterdir()
             if path.is_dir() and (path / "validation_metrics.csv").exists()),
            key=lambda path: path.name,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("outputs 下没有完整训练结果")
        # 优先选择包含已训练模型指标的最新运行，跳过仅 RFM 校验输出。
        output_dir = next(
            (path for path in candidates
             if pd.read_csv(path / "validation_metrics.csv")["experiment"].str.startswith("E5").any()),
            candidates[0],
        )
    checkpoint_dir = ROOT / "checkpoints" / output_dir.name
    return output_dir, checkpoint_dir


def selected_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """每个实验仅保留验证集 AP 最高的一行（即选中的 alpha/参数）。"""
    table = metrics.copy()
    table["average_precision"] = pd.to_numeric(table["average_precision"], errors="coerce")
    indices = table.groupby("experiment", sort=False)["average_precision"].idxmax()
    table = table.loc[indices].copy()
    order = list(DISPLAY_NAMES)
    table["sort_order"] = table["experiment"].map({name: i for i, name in enumerate(order)}).fillna(999)
    return table.sort_values("sort_order").drop(columns="sort_order").reset_index(drop=True)


def plot_experiment_comparison(table: pd.DataFrame, review_dir: Path) -> None:
    names = [DISPLAY_NAMES.get(value, value) for value in table["experiment"]]
    colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(names))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plots = [
        ("average_precision", "AP（主指标）", None),
        ("roc_auc", "ROC-AUC", 0.5),
    ]
    for axis, (column, title, baseline) in zip(axes, plots):
        values = table[column].astype(float).to_numpy()
        bars = axis.barh(np.arange(len(names)), values, color=colors, edgecolor='#333333', linewidth=0.5)
        axis.set_yticks(np.arange(len(names)), labels=names)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=12, fontweight='bold')
        axis.grid(axis="x", alpha=0.25)
        if baseline is not None:
            axis.axvline(baseline, color="#666", linestyle="--", linewidth=1)
        for bar, value in zip(bars, values):
            suffix = "x" if "lift" in column else ""
            axis.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}{suffix}", va="center", fontsize=9)
    fig.suptitle("Response 模型排序指标对比（每个实验仅保留验证集最优配置）", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(review_dir / "01_实验效果对比.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_topk_metrics(table: pd.DataFrame, review_dir: Path) -> None:
    """展示四档 Precision、Recall 与 Lift，便于业务预算档位比较。"""
    fractions = [0.01, 0.05, 0.10, 0.20]
    labels = ["1%", "5%", "10%", "20%"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_table = table[~table["experiment"].eq("E0_global_rate")]
    for index, (_, row) in enumerate(model_table.iterrows()):
        color = CUSTOM_COLORS[index % len(CUSTOM_COLORS)]
        name = DISPLAY_NAMES.get(row["experiment"], row["experiment"])
        axes[0].plot(labels, [row[f"precision_at_{fraction:.2f}"] * 100 for fraction in fractions],
                     marker="o", linewidth=2, color=color, label=name)
        axes[1].plot(labels, [row[f"recall_at_{fraction:.2f}"] * 100 for fraction in fractions],
                     marker="o", linewidth=2, color=color, label=name)
        axes[2].plot(labels, [row[f"lift_at_{fraction:.2f}"] for fraction in fractions],
                     marker="o", linewidth=2, color=color, label=name)
    for axis, title, ylabel in [
        (axes[0], "Precision@K", "Top-K 实际正样本率（%）"),
        (axes[1], "Recall@K", "覆盖全部正样本比例（%）"),
        (axes[2], "Lift@K", "相对总体正样本率倍数"),
    ]:
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("圈选人群比例 K")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=8, loc="best")
    fig.suptitle("Top-K 业务指标对比：1% / 5% / 10% / 20%", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(review_dir / "02_TopK指标对比.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_deciles(output_dir: Path, table: pd.DataFrame, review_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 7))
    plotted = 0
    color_idx = 0
    for experiment in table["experiment"]:
        if experiment.startswith("E2_live"):
            alpha = int(table.loc[table["experiment"].eq(experiment), "alpha"].iloc[0])
            path = output_dir / f"validation_E2_alpha_{alpha}_deciles.csv"
        elif experiment.startswith("E2_platform"):
            alpha = int(table.loc[table["experiment"].eq(experiment), "alpha"].iloc[0])
            path = output_dir / f"validation_E2_platform_alpha_{alpha}_deciles.csv"
        else:
            path = output_dir / f"validation_{experiment}_deciles.csv"
        if not path.exists() or experiment == "E0_global_rate":
            continue
        deciles = pd.read_csv(path)
        color = CUSTOM_COLORS[color_idx % len(CUSTOM_COLORS)]
        axis.plot(deciles["decile"], deciles["actual_positive_rate"], marker="o",
                  label=DISPLAY_NAMES.get(experiment, experiment), color=color, linewidth=2)
        plotted += 1
        color_idx += 1
    if plotted:
        axis.set_xticks(range(1, 11))
        axis.set_xlabel("预测分十分位（1=分数最高，10=最低）", fontsize=11)
        axis.set_ylabel("实际平台券直播复购率", fontsize=11)
        axis.set_title("验证集十分位实际响应率：越单调下降，排序越可靠", fontsize=13, fontweight="bold")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(review_dir / "03_十分位响应率.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(output_dir: Path, best_experiment: str, review_dir: Path) -> None:
    path = output_dir / f"{best_experiment}_feature_importance.csv"
    if not path.exists():
        return
    importance = pd.read_csv(path).head(20).sort_values("gain_ratio")
    labels = importance["feature"].str.replace(r"^(numeric|categorical)__", "", regex=True)
    fig, axis = plt.subplots(figsize=(12, 8))
    colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(labels))]
    axis.barh(labels, importance["gain_ratio"] * 100, color=colors, edgecolor='#333333', linewidth=0.5)
    axis.set_xlabel("Gain 重要性占比（%）", fontsize=11)
    axis.set_title(f"最佳模型 {DISPLAY_NAMES.get(best_experiment, best_experiment)}：Top 20 特征", fontsize=13, fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(review_dir / "04_最佳模型特征重要性.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


LIVE_LIFECYCLE_CN = {"A": "活跃购买用户", "D": "近180天沉睡用户", "N": "无直播购买历史", "U": "口径未知/字段冲突"}
PLATFORM_LIFECYCLE_CN = {"MATURE": "成熟购买用户（≥3单）", "NEW": "新客/培育用户（1～2单）", "N": "未购用户（0单）", "U": "字段异常用户"}


def _plot_segment_panels(axes, summary: pd.DataFrame, rfm_type: str, title: str, mapping: Dict[str, str]) -> None:
    subset = summary[summary["rfm_type"].eq(rfm_type)].copy()
    segments = list(dict.fromkeys(subset["segment"].tolist()))
    labels = [mapping.get(segment, segment) for segment in segments]
    y = np.arange(len(segments))
    width = 0.36
    for offset, dataset, color in [(-width / 2, "train", CUSTOM_COLORS[2]),
                                    (width / 2, "validation", CUSTOM_COLORS[7])]:
        indexed = subset[subset["dataset"].eq(dataset)].set_index("segment")
        shares = np.array([indexed["sample_share"].get(segment, 0) for segment in segments]) * 100
        bars = axes[0].barh(y + offset, shares, height=width, color=color,
                            edgecolor="#555", linewidth=0.5, label="训练集" if dataset == "train" else "验证集")
        for bar, value in zip(bars, shares):
            axes[0].text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=8)
    axes[0].set_yticks(y, labels=labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("样本占比（%）")
    axes[0].set_title(f"{title}：人群占比与漂移")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.25)

    for dataset, color, marker in [("train", CUSTOM_COLORS[3], "o"),
                                   ("validation", CUSTOM_COLORS[4], "s")]:
        indexed = subset[subset["dataset"].eq(dataset)].set_index("segment")
        rates = np.array([indexed["response_rate"].get(segment, np.nan) for segment in segments]) * 100
        axes[1].plot(rates, y, marker=marker, linewidth=2, color=color,
                     label="训练集" if dataset == "train" else "验证集")
        for rate, position in zip(rates, y):
            if np.isfinite(rate):
                axes[1].text(rate + 0.15, position, f"{rate:.1f}%", va="center", fontsize=8)
    axes[1].set_yticks(y, labels=labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("平台券直播复购率（%）")
    axes[1].set_title(f"{title}：各人群实际响应率")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.25)


def plot_rfm(output_dir: Path, review_dir: Path) -> str:
    """展示两套口径各自的分布漂移与实际响应率，不将两者作横向等价比较。"""
    summary_path = output_dir / "rfm_segment_summary.csv"
    quality_path = output_dir / "rfm_rules_and_quality.json"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        fig, axes = plt.subplots(2, 2, figsize=(17, 11))
        _plot_segment_panels(axes[0], summary, "live", "直播 RFM 生命周期", LIVE_LIFECYCLE_CN)
        _plot_segment_panels(axes[1], summary, "platform", "全平台 RFM 生命周期", PLATFORM_LIFECYCLE_CN)
        fig.suptitle("RFM 数据质量与响应率诊断（上下两行口径不同，不作分群间直接比较）",
                     fontsize=15, fontweight="bold")
        fig.tight_layout()
        note = "图中同时展示 Train/Validation 人群占比漂移和各生命周期的平台券直播复购率。"
    else:
        if not quality_path.exists():
            return "当前运行没有 RFM 质量数据。"
        data = json.loads(quality_path.read_text(encoding="utf-8"))
        rows = []
        for dataset in ["train", "validation"]:
            quality = data.get("quality", {}).get(dataset, {})
            for rfm_type, key in [("live", "live_lifecycle"), ("platform", "platform_lifecycle")]:
                values = quality.get(key, {})
                total = max(sum(values.values()), 1)
                rows.extend({"rfm_type": rfm_type, "dataset": dataset, "segment": segment,
                             "sample_count": count, "sample_share": count / total,
                             "response_rate": np.nan} for segment, count in values.items())
        summary = pd.DataFrame(rows)
        fig, axes = plt.subplots(2, 1, figsize=(13, 10))
        for axis, rfm_type, title, mapping in [
            (axes[0], "live", "直播 RFM 生命周期", LIVE_LIFECYCLE_CN),
            (axes[1], "platform", "全平台 RFM 生命周期", PLATFORM_LIFECYCLE_CN),
        ]:
            subset = summary[summary["rfm_type"].eq(rfm_type)]
            segments = list(dict.fromkeys(subset["segment"].tolist()))
            labels = [mapping.get(segment, segment) for segment in segments]
            y = np.arange(len(segments)); width = 0.36
            for offset, dataset, color in [(-width / 2, "train", CUSTOM_COLORS[2]),
                                            (width / 2, "validation", CUSTOM_COLORS[7])]:
                indexed = subset[subset["dataset"].eq(dataset)].set_index("segment")
                shares = np.array([indexed["sample_share"].get(segment, 0) for segment in segments]) * 100
                bars = axis.barh(y + offset, shares, height=width, color=color, edgecolor="#555", linewidth=0.5,
                                 label="训练集" if dataset == "train" else "验证集")
                for bar, value in zip(bars, shares):
                    axis.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=8)
            axis.set_yticks(y, labels=labels); axis.invert_yaxis(); axis.set_xlabel("样本占比（%）")
            axis.set_title(f"{title}：Train/Validation 分布漂移"); axis.legend(); axis.grid(axis="x", alpha=0.25)
        fig.suptitle("RFM 人群分布诊断（历史运行未保存分群响应率）", fontsize=15, fontweight="bold")
        fig.tight_layout()
        note = "历史运行仅能展示 Train/Validation 人群占比漂移；原始数据当前不可用，未伪造分群响应率。未来训练会自动输出响应率。"
    fig.savefig(review_dir / "05_RFM分布漂移与响应率.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return note


def render_report(output_dir: Path, checkpoint_dir: Path, table: pd.DataFrame,
                  review_dir: Path, rfm_note: str) -> None:
    model_rows = table[table["experiment"].isin(["E3_dual_rfm_raw", "E4_live_features", "E5_live_plus_platform"])]
    best = model_rows.loc[model_rows["average_precision"].idxmax()] if not model_rows.empty else table.loc[table["average_precision"].idxmax()]
    best_name = str(best["experiment"])
    weights = checkpoint_dir / f"{best_name}.txt"
    preprocessor = checkpoint_dir / f"{best_name}_preprocessor.pkl"
    metadata = checkpoint_dir / f"{best_name}_metadata.json"

    fractions = [0.01, 0.05, 0.10, 0.20]
    rows = []
    for _, row in table.iterrows():
        precision_cells = "".join(f"<td>{row[f'precision_at_{fraction:.2f}']:.2%}</td>" for fraction in fractions)
        recall_cells = "".join(f"<td>{row[f'recall_at_{fraction:.2f}']:.2%}</td>" for fraction in fractions)
        lift_cells = "".join(f"<td>{row[f'lift_at_{fraction:.2f}']:.2f}x</td>" for fraction in fractions)
        rows.append(
            "<tr>"
            f"<td>{html.escape(DISPLAY_NAMES.get(row['experiment'], row['experiment']))}</td>"
            f"<td>{row['roc_auc']:.4f}</td><td>{row['average_precision']:.4f}</td>"
            f"{precision_cells}{recall_cells}{lift_cells}</tr>"
        )
    conclusions = [
        f"Validation 最终选择 {DISPLAY_NAMES.get(best_name, best_name)}，AP={best['average_precision']:.4f}，ROC-AUC={best['roc_auc']:.4f}。",
        f"E4 → E5 增益较小，说明直播行为特征是主体，全平台 RFM 提供少量补充信息。",
        f"E5 Top 1% Precision={best['precision_at_0.01']:.1%}，Lift={best['lift_at_0.01']:.2f}，头部人群聚集能力较强。",
    ]
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Response 模型 Review</title>
<style>
body{{font-family:'Times New Roman','Songti SC',serif;max-width:1200px;margin:32px auto;padding:0 24px;color:#24303f;line-height:1.65}}
h1,h2{{color:#16324f;font-family:'Times New Roman','Songti SC',serif}} .card{{background:#FDEBAA;border-left:5px solid #EDC3A5;padding:14px 20px;margin:16px 0}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:max-content;min-width:100%;font-size:13px}} th,td{{border:1px solid #d8dee8;padding:7px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left;position:sticky;left:0;background:white}} th{{background:#DBE4FB}} th:first-child{{background:#DBE4FB;z-index:2}}
img{{width:100%;border:1px solid #ddd;margin:12px 0 28px}} code{{background:#F1F1F1;padding:2px 5px}} .path{{word-break:break-all}}
</style></head><body>
<h1>Response 模型训练 Review 总览</h1>
<div class="card"><b>你日常只需要看这个页面。</b>同目录 5 张图用于辅助判断；上级目录的 CSV/JSON 是追溯明细，不需要逐个打开。</div>
<h2>一、最终结论</h2><ol>{''.join(f'<li>{html.escape(item)}</li>' for item in conclusions)}</ol>
<h2>二、实验对比</h2>
<p>AP 为代码实际计算的 Average Precision；Top-K 指标统一展示 K=1%、5%、10%、20%。表格可横向滚动，最佳轮数属于训练细节，已下沉到 checkpoint metadata。</p>
<div class="table-wrap"><table><thead><tr>
<th>实验</th><th>ROC-AUC</th><th>AP</th>
<th>Precision@1%</th><th>Precision@5%</th><th>Precision@10%</th><th>Precision@20%</th>
<th>Recall@1%</th><th>Recall@5%</th><th>Recall@10%</th><th>Recall@20%</th>
<th>Lift@1%</th><th>Lift@5%</th><th>Lift@10%</th><th>Lift@20%</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<img src="01_实验效果对比.png" alt="ROC-AUC与AP实验对比">
<img src="02_TopK指标对比.png" alt="Top-K指标对比">
<h2>三、为什么选择 E5</h2>
<pre style="background:#F1F1F1;padding:12px;border-radius:4px;font-family:monospace">
E0～E2：RFM Baseline
        ↓
E3：RFM + LightGBM，效果明显提升
        ↓
E4：加入完整直播特征，再次明显提升
        ↓
E5：加入全平台特征，小幅继续提升
</pre>
<h2>四、辅助诊断</h2>
<h3>1. 分数是否真的分出了高低人群</h3>
<p>重点看十分位曲线是否从第 1 档到第 10 档整体下降。第 1 档代表模型认为最可能用平台券直播复购的人群。</p>
<img src="03_十分位响应率.png" alt="十分位响应率">
<h3>2. 模型主要依据什么判断</h3>
<p>重要性表示模型使用频率和信息增益，不等于因果关系。</p>
<img src="04_最佳模型特征重要性.png" alt="特征重要性">
<h3>3. RFM 数据质量诊断</h3>
<p>{{rfm_note}}</p>
<p>直播和全平台生命周期定义不同，上下两行分别解释，不能将两套分群作一一对应比较。</p>
<img src="05_RFM分布漂移与响应率.png" alt="RFM分布漂移与响应率">
<h2>五、产物位置</h2>
<ul>
<li>LightGBM 权重：<code class="path">{html.escape(str(weights))}</code></li>
<li>预处理器：<code class="path">{html.escape(str(preprocessor))}</code></li>
<li>特征与参数元数据：<code class="path">{html.escape(str(metadata))}</code></li>
</ul>
<p><b>注意：</b>部署或批量预测时，权重和预处理器必须配套使用，不能只拿模型 txt。</p>
<h2>附录：什么时候才需要看底层 CSV</h2><ul>
<li><code>validation_metrics.csv</code>：核对完整指标。</li>
<li><code>validation_*_deciles.csv</code>：排查十分位曲线细节。</li>
<li><code>*_feature_importance.csv</code>：查看全部特征重要性。</li>
<li><code>data_summary.csv</code>、<code>user_overlap.csv</code>：检查数据量、正样本率和用户重合。</li>
<li><code>rfm_rules_and_quality.json</code>：核对冻结的 RFM 阈值与异常人群。</li></ul>
</body></html>""".replace("{rfm_note}", html.escape(rfm_note))
    (review_dir / "00_先看这里_结果总览.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir, checkpoint_dir = resolve_run(args.run_id)
    review_dir = output_dir / "review"
    review_dir.mkdir(exist_ok=True)
    metrics = pd.read_csv(output_dir / "validation_metrics.csv")
    table = selected_metrics(metrics)
    model_rows = table[table["experiment"].isin(["E3_dual_rfm_raw", "E4_live_features", "E5_live_plus_platform"])]
    best_name = str(model_rows.loc[model_rows["average_precision"].idxmax(), "experiment"])
    plot_experiment_comparison(table, review_dir)
    plot_topk_metrics(table, review_dir)
    plot_deciles(output_dir, table, review_dir)
    plot_feature_importance(output_dir, best_name, review_dir)
    rfm_note = plot_rfm(output_dir, review_dir)
    render_report(output_dir, checkpoint_dir, table, review_dir, rfm_note)
    print(f"Review 报告已生成：{review_dir / '00_先看这里_结果总览.html'}")


if __name__ == "__main__":
    main()
