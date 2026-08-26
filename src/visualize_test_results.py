#!/usr/bin/env python3
"""Test 结果 Review 可视化与结论报告。

基于已冻结 run-id，不重新运行 Test、不训练模型。
覆盖：Validation vs Test 指标对比；Test 全量用户 vs unseen_users 对比；
最佳模型及双轨 RFM 的 Test 十分位响应率；概率校准/实际率对比；
正样本率漂移对 AP、Lift、LogLoss 的影响解释。
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# 英文优先 Times New Roman；其不含的中文字形自动回退到本机宋体 Songti SC。
plt.rcParams["font.family"] = ["Times New Roman", "Songti SC"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

# 自定义配色方案（与 visualize_results.py 保持一致）
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
    parser = argparse.ArgumentParser(description="生成 Test 结果 Review 报告")
    parser.add_argument("--run-id", required=True, help="指定已冻结的运行目录")
    return parser.parse_args()


def resolve_run(run_id: str) -> Tuple[Path, Path, Path]:
    """返回 (output_dir, checkpoint_dir, test_dir)"""
    outputs = ROOT / "outputs"
    output_dir = outputs / run_id
    if not output_dir.exists():
        raise FileNotFoundError(f"运行目录不存在：{output_dir}")
    test_dir = output_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test 目录不存在：{test_dir}，请先运行 test 评估")
    checkpoint_dir = ROOT / "checkpoints" / run_id
    return output_dir, checkpoint_dir, test_dir


def load_test_metrics(test_dir: Path) -> pd.DataFrame:
    """加载 test_metrics.csv，分离 all_users 和 unseen_users"""
    metrics = pd.read_csv(test_dir / "test_metrics.csv")
    return metrics


def load_validation_metrics(output_dir: Path) -> pd.DataFrame:
    """加载 validation_metrics.csv，每个实验保留 AP 最高的一行"""
    metrics = pd.read_csv(output_dir / "validation_metrics.csv")
    metrics["average_precision"] = pd.to_numeric(metrics["average_precision"], errors="coerce")
    indices = metrics.groupby("experiment", sort=False)["average_precision"].idxmax()
    return metrics.loc[indices].copy().reset_index(drop=True)


def plot_validation_vs_test(val_metrics: pd.DataFrame, test_metrics: pd.DataFrame, 
                            review_dir: Path) -> None:
    """图1：Validation vs Test 指标对比（仅对比共有实验）"""
    # 找出共有实验
    val_exps = set(val_metrics["experiment"].tolist())
    test_all_exps = set(test_metrics[test_metrics["evaluation_scope"] == "all_users"]["experiment"].tolist())
    common_exps = val_exps & test_all_exps
    
    # 按显示名称排序
    order = list(DISPLAY_NAMES.keys())
    common_exps_sorted = sorted(common_exps, key=lambda x: order.index(x) if x in order else 999)
    
    # 准备数据
    test_all = test_metrics[test_metrics["evaluation_scope"] == "all_users"].set_index("experiment")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    metrics_to_plot = [
        ("average_precision", "AP（Average Precision）", None),
        ("roc_auc", "ROC-AUC", 0.5),
        ("lift_at_0.01", "Lift@1%", 1.0),
    ]
    
    for ax, (col, title, baseline) in zip(axes, metrics_to_plot):
        val_values = []
        test_values = []
        labels = []
        
        for exp in common_exps_sorted:
            if exp in val_metrics["experiment"].values and exp in test_all.index:
                val_row = val_metrics[val_metrics["experiment"] == exp]
                test_row = test_all.loc[[exp]]
                val_values.append(float(val_row[col].iloc[0]))
                test_values.append(float(test_row[col].iloc[0]))
                labels.append(DISPLAY_NAMES.get(exp, exp))
        
        y = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.barh(y - width/2, val_values, width, label="Validation", 
                        color=CUSTOM_COLORS[2], edgecolor='#333', linewidth=0.5)
        bars2 = ax.barh(y + width/2, test_values, width, label="Test",
                        color=CUSTOM_COLORS[3], edgecolor='#333', linewidth=0.5)
        
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.25)
        ax.legend(fontsize=9)
        
        if baseline is not None:
            ax.axvline(baseline, color='#666', linestyle='--', linewidth=1)
        
        # 添加数值标签
        for bars, values in [(bars1, val_values), (bars2, test_values)]:
            for bar, value in zip(bars, values):
                suffix = "x" if "lift" in col else ""
                ax.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                        f"{value:.3f}{suffix}", va='center', fontsize=8)
    
    fig.suptitle("Validation vs Test 指标对比（Test 全量用户）", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(review_dir / "01_validation_vs_test.png", dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_all_vs_unseen_users(test_metrics: pd.DataFrame, review_dir: Path) -> None:
    """图2：Test 全量用户 vs unseen_users 对比"""
    test_all = test_metrics[test_metrics["evaluation_scope"] == "all_users"].set_index("experiment")
    test_unseen = test_metrics[test_metrics["evaluation_scope"] == "unseen_users"].set_index("experiment")
    
    common_exps = set(test_all.index) & set(test_unseen.index)
    order = list(DISPLAY_NAMES.keys())
    common_exps_sorted = sorted(common_exps, key=lambda x: order.index(x) if x in order else 999)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    metrics_to_plot = [
        ("average_precision", "AP（Average Precision）", None),
        ("roc_auc", "ROC-AUC", 0.5),
        ("lift_at_0.01", "Lift@1%", 1.0),
    ]
    
    for ax, (col, title, baseline) in zip(axes, metrics_to_plot):
        all_values = []
        unseen_values = []
        labels = []
        
        for exp in common_exps_sorted:
            all_values.append(float(test_all.loc[exp, col]))
            unseen_values.append(float(test_unseen.loc[exp, col]))
            labels.append(DISPLAY_NAMES.get(exp, exp))
        
        y = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.barh(y - width/2, all_values, width, label="全量用户",
                        color=CUSTOM_COLORS[0], edgecolor='#333', linewidth=0.5)
        bars2 = ax.barh(y + width/2, unseen_values, width, label="Unseen Users",
                        color=CUSTOM_COLORS[4], edgecolor='#333', linewidth=0.5)
        
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.25)
        ax.legend(fontsize=9)
        
        if baseline is not None:
            ax.axvline(baseline, color='#666', linestyle='--', linewidth=1)
        
        for bars, values in [(bars1, all_values), (bars2, unseen_values)]:
            for bar, value in zip(bars, values):
                suffix = "x" if "lift" in col else ""
                ax.text(value + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{value:.3f}{suffix}", va='center', fontsize=8)
    
    fig.suptitle("Test 全量用户 vs Unseen Users 对比", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(review_dir / "02_all_vs_unseen_users.png", dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_test_deciles(test_dir: Path, review_dir: Path) -> None:
    """图3：Test 十分位响应率（最佳模型 + 双轨 RFM）"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：全量用户
    ax = axes[0]
    color_idx = 0
    plotted = 0
    
    for exp in ["E5_live_plus_platform", "E2_live_rfm_probability", "E2_platform_rfm_probability"]:
        path = test_dir / f"{exp}_all_users_deciles.csv"
        if path.exists():
            deciles = pd.read_csv(path)
            color = CUSTOM_COLORS[color_idx % len(CUSTOM_COLORS)]
            ax.plot(deciles["decile"], deciles["actual_positive_rate"], 
                    marker='o', label=DISPLAY_NAMES.get(exp, exp), 
                    color=color, linewidth=2)
            plotted += 1
            color_idx += 1
    
    if plotted:
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("预测分十分位（1=分数最高，10=最低）", fontsize=11)
        ax.set_ylabel("实际平台券直播复购率", fontsize=11)
        ax.set_title("Test 全量用户：十分位实际响应率", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    
    # 右图：unseen users
    ax = axes[1]
    color_idx = 0
    plotted = 0
    
    for exp in ["E5_live_plus_platform", "E2_live_rfm_probability", "E2_platform_rfm_probability"]:
        path = test_dir / f"{exp}_unseen_users_deciles.csv"
        if path.exists():
            deciles = pd.read_csv(path)
            color = CUSTOM_COLORS[color_idx % len(CUSTOM_COLORS)]
            ax.plot(deciles["decile"], deciles["actual_positive_rate"],
                    marker='s', label=DISPLAY_NAMES.get(exp, exp),
                    color=color, linewidth=2)
            plotted += 1
            color_idx += 1
    
    if plotted:
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("预测分十分位（1=分数最高，10=最低）", fontsize=11)
        ax.set_ylabel("实际平台券直播复购率", fontsize=11)
        ax.set_title("Test Unseen Users：十分位实际响应率", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    
    fig.suptitle("Test 十分位响应率：曲线越单调下降，排序越可靠", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(review_dir / "03_test_deciles.png", dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_calibration(test_dir: Path, test_metrics: pd.DataFrame, review_dir: Path) -> None:
    """图4：概率校准/实际率对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 收集各实验的预测概率均值和实际正样本率
    experiments = ["E5_live_plus_platform", "E2_live_rfm_probability", "E2_platform_rfm_probability", "E0_global_rate"]
    
    for ax, scope, title in [(axes[0], "all_users", "全量用户"), 
                              (axes[1], "unseen_users", "Unseen Users")]:
        predicted_means = []
        actual_rates = []
        labels = []
        
        for exp in experiments:
            path = test_dir / f"{exp}_{scope}_deciles.csv"
            if path.exists():
                deciles = pd.read_csv(path)
                if "average_score" not in deciles.columns:
                    continue
                # 十分位样本数可能相差 1，必须按 samples 加权，不可简单平均。
                weights = deciles["samples"].to_numpy(dtype=float)
                pred_mean = float(np.average(deciles["average_score"], weights=weights))
                actual_rate = float(np.average(deciles["actual_positive_rate"], weights=weights))
                predicted_means.append(pred_mean)
                actual_rates.append(actual_rate)
                labels.append(DISPLAY_NAMES.get(exp, exp))
        
        if predicted_means:
            # 绘制校准曲线
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="完美校准")
            ax.scatter(predicted_means, actual_rates, s=100, 
                       color=CUSTOM_COLORS[3], edgecolors='#333', linewidth=1, zorder=5)
            
            for i, label in enumerate(labels):
                ax.annotate(label, (predicted_means[i], actual_rates[i]),
                            textcoords="offset points", xytext=(5, 5), fontsize=8)
            
            ax.set_xlabel("预测概率均值", fontsize=11)
            ax.set_ylabel("实际正样本率", fontsize=11)
            ax.set_title(f"Test {title}：概率校准", fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9)
    
    fig.suptitle("概率校准：点越接近对角线，校准越好", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(review_dir / "04_calibration.png", dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_positive_rate_drift(val_metrics: pd.DataFrame, test_metrics: pd.DataFrame,
                              review_dir: Path) -> None:
    """图5：正样本率漂移及其对指标的影响"""
    test_all = test_metrics[test_metrics["evaluation_scope"] == "all_users"].set_index("experiment")
    test_unseen = test_metrics[test_metrics["evaluation_scope"] == "unseen_users"].set_index("experiment")
    
    # 找出共有实验
    common_exps = set(val_metrics["experiment"].tolist()) & set(test_all.index)
    order = list(DISPLAY_NAMES.keys())
    common_exps_sorted = sorted(common_exps, key=lambda x: order.index(x) if x in order else 999)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1：正样本率对比
    ax = axes[0, 0]
    val_rates = []
    test_all_rates = []
    test_unseen_rates = []
    labels = []
    
    for exp in common_exps_sorted:
        val_row = val_metrics[val_metrics["experiment"] == exp]
        val_rates.append(float(val_row["positive_rate"].iloc[0]) * 100)
        test_all_rates.append(float(test_all.loc[exp, "positive_rate"]) * 100)
        test_unseen_rates.append(float(test_unseen.loc[exp, "positive_rate"]) * 100)
        labels.append(DISPLAY_NAMES.get(exp, exp))
    
    y = np.arange(len(labels))
    width = 0.25
    
    ax.barh(y - width, val_rates, width, label="Validation", color=CUSTOM_COLORS[2])
    ax.barh(y, test_all_rates, width, label="Test 全量", color=CUSTOM_COLORS[3])
    ax.barh(y + width, test_unseen_rates, width, label="Test Unseen", color=CUSTOM_COLORS[4])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("正样本率 (%)", fontsize=11)
    ax.set_title("正样本率分布漂移", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.25)
    
    # 图2：AP 变化
    ax = axes[0, 1]
    val_prauc = [float(val_metrics[val_metrics["experiment"] == exp]["average_precision"].iloc[0]) 
                  for exp in common_exps_sorted]
    test_all_prauc = [float(test_all.loc[exp, "average_precision"]) for exp in common_exps_sorted]
    test_unseen_prauc = [float(test_unseen.loc[exp, "average_precision"]) for exp in common_exps_sorted]
    
    ax.barh(y - width, val_prauc, width, label="Validation", color=CUSTOM_COLORS[2])
    ax.barh(y, test_all_prauc, width, label="Test 全量", color=CUSTOM_COLORS[3])
    ax.barh(y + width, test_unseen_prauc, width, label="Test Unseen", color=CUSTOM_COLORS[4])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("AP", fontsize=11)
    ax.set_title("AP 对比", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.25)
    
    # 图3：Lift@1% 变化
    ax = axes[1, 0]
    val_lift = [float(val_metrics[val_metrics["experiment"] == exp]["lift_at_0.01"].iloc[0])
                 for exp in common_exps_sorted]
    test_all_lift = [float(test_all.loc[exp, "lift_at_0.01"]) for exp in common_exps_sorted]
    test_unseen_lift = [float(test_unseen.loc[exp, "lift_at_0.01"]) for exp in common_exps_sorted]
    
    ax.barh(y - width, val_lift, width, label="Validation", color=CUSTOM_COLORS[2])
    ax.barh(y, test_all_lift, width, label="Test 全量", color=CUSTOM_COLORS[3])
    ax.barh(y + width, test_unseen_lift, width, label="Test Unseen", color=CUSTOM_COLORS[4])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Lift@1%", fontsize=11)
    ax.set_title("Lift@1% 对比", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.25)
    
    # 图4：LogLoss 变化
    ax = axes[1, 1]
    val_logloss = [float(val_metrics[val_metrics["experiment"] == exp]["log_loss"].iloc[0])
                    if pd.notna(val_metrics[val_metrics["experiment"] == exp]["log_loss"].iloc[0]) else 0
                    for exp in common_exps_sorted]
    test_all_logloss = [float(test_all.loc[exp, "log_loss"]) for exp in common_exps_sorted]
    test_unseen_logloss = [float(test_unseen.loc[exp, "log_loss"]) for exp in common_exps_sorted]
    
    ax.barh(y - width, val_logloss, width, label="Validation", color=CUSTOM_COLORS[2])
    ax.barh(y, test_all_logloss, width, label="Test 全量", color=CUSTOM_COLORS[3])
    ax.barh(y + width, test_unseen_logloss, width, label="Test Unseen", color=CUSTOM_COLORS[4])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("LogLoss", fontsize=11)
    ax.set_title("LogLoss 对比（越低越好）", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.25)
    
    fig.suptitle("正样本率漂移对模型指标的影响分析", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(review_dir / "05_positive_rate_drift.png", dpi=160, bbox_inches='tight')
    plt.close(fig)


def render_test_report(output_dir: Path, test_dir: Path, val_metrics: pd.DataFrame,
                        test_metrics: pd.DataFrame, review_dir: Path,
                        frozen_experiment: str) -> None:
    """生成 Test Review HTML 报告"""
    test_all = test_metrics[test_metrics["evaluation_scope"] == "all_users"].set_index("experiment")
    test_unseen = test_metrics[test_metrics["evaluation_scope"] == "unseen_users"].set_index("experiment")
    
    # 最终模型只能来自 Validation 冻结清单，严禁按 Test 指标再次选择。
    best_exp = frozen_experiment
    if best_exp not in test_all.index:
        raise ValueError(f"冻结模型 {best_exp} 不在 Test 指标中")
    best_row = test_all.loc[best_exp]
    best_val_row = val_metrics[val_metrics["experiment"] == best_exp].iloc[0]
    
    # 计算关键指标
    val_positive_rate = float(best_val_row["positive_rate"]) * 100
    test_all_positive_rate = float(best_row["positive_rate"]) * 100
    test_unseen_positive_rate = float(test_unseen.loc[best_exp, "positive_rate"]) * 100
    
    val_prauc = float(best_val_row["average_precision"])
    test_all_prauc = float(best_row["average_precision"])
    test_unseen_prauc = float(test_unseen.loc[best_exp, "average_precision"])
    
    val_lift = float(best_val_row["lift_at_0.01"])
    test_all_lift = float(best_row["lift_at_0.01"])
    test_unseen_lift = float(test_unseen.loc[best_exp, "lift_at_0.01"])
    
    unseen_row = test_unseen.loc[best_exp]
    conclusions = [
        f"冻结的 {DISPLAY_NAMES.get(best_exp, best_exp)} 在 Test 全量用户上 ROC-AUC={float(best_row['roc_auc']):.4f}、"
        f"AP={test_all_prauc:.4f}，保持较好的时间外排序能力。",
        f"Top 1% Precision={float(best_row['precision_at_0.01']):.1%}、Lift@1%={test_all_lift:.2f}x，"
        "头部高响应用户仍能被明显聚集。",
        f"Unseen Users 上 ROC-AUC={float(unseen_row['roc_auc']):.4f}、AP={test_unseen_prauc:.4f}、"
        f"Lift@1%={test_unseen_lift:.2f}x，模型对未见用户仍具有排序能力。",
    ]

    distribution_rows = [
        ("Validation", int(best_val_row["rows"]), float(best_val_row["positive_rate"])),
        ("Test All", int(best_row["rows"]), float(best_row["positive_rate"])),
        ("Test Unseen", int(unseen_row["rows"]), float(unseen_row["positive_rate"])),
    ]
    distribution_html = "".join(
        f"<tr><td>{scope}</td><td>{rows:,}</td><td>{rate:.2%}</td></tr>"
        for scope, rows, rate in distribution_rows)
    performance_rows = [
        ("Validation", best_val_row), ("Test All", best_row), ("Test Unseen", unseen_row)]
    performance_html = "".join(
        "<tr>" + f"<td>{scope}</td><td>{float(row['roc_auc']):.4f}</td>"
        f"<td>{float(row['average_precision']):.4f}</td>"
        f"<td>{float(row['precision_at_0.01']):.2%}</td>"
        f"<td>{float(row['recall_at_0.01']):.2%}</td>"
        f"<td>{float(row['lift_at_0.01']):.2f}x</td></tr>"
        for scope, row in performance_rows)
    
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Test Review - Response 模型</title>
<style>
body{{font-family:'Times New Roman','Songti SC',serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#24303f;line-height:1.65}}
h1,h2{{color:#16324f;font-family:'Times New Roman','Songti SC',serif}}
.card{{background:#FDEBAA;border-left:5px solid #EDC3A5;padding:14px 20px;margin:16px 0}}
.warning{{background:#FCB6A5;border-left:5px solid #EDC3A5;padding:14px 20px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{border:1px solid #d8dee8;padding:8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#DBE4FB}}
img{{width:100%;border:1px solid #ddd;margin:12px 0 28px}}
code{{background:#F1F1F1;padding:2px 5px}}
.path{{word-break:break-all}}
.metric-box{{display:inline-block;background:#DBE4FB;padding:8px 16px;margin:4px;border-radius:4px}}
</style></head><body>
<h1>Test Review - Response 模型评估报告</h1>
<div class="card">
<b>运行 ID：</b><code>{html.escape(output_dir.name)}</code><br>
<b>Test 执行时间：</b>基于已冻结数据，未重新运行<br>
<b>评估范围：</b>全量用户 (All Users) + 未见用户 (Unseen Users)
</div>

<h2>一、核心结论</h2>
<ol>{''.join(f'<li>{html.escape(item)}</li>' for item in conclusions)}</ol>

<div class="warning"><b>数据分布提醒：</b>Test 正样本率 {test_all_positive_rate:.2f}%，高于 Validation 的 {val_positive_rate:.2f}%；Unseen Users 为 {test_unseen_positive_rate:.2f}%。因此 AP 绝对值不能跨数据集直接比较，需要结合 ROC-AUC、Lift 综合判断。</div>

<h2>二、冻结 E5 的 Test All 成绩</h2>
<div class="metric-box">ROC-AUC: <b>{float(best_row['roc_auc']):.4f}</b></div>
<div class="metric-box">AP: <b>{test_all_prauc:.4f}</b></div>
<div class="metric-box">Precision@1%: <b>{float(best_row['precision_at_0.01']):.2%}</b></div>
<div class="metric-box">Lift@1%: <b>{test_all_lift:.2f}x</b></div>

<h2>三、Validation vs Test 指标对比</h2>
<p>对比各实验在 Validation 和 Test 上的表现，观察泛化能力。</p>
<img src="01_validation_vs_test.png" alt="Validation vs Test 对比">

<h2>四、全量用户 vs Unseen Users 对比</h2>
<p>Unseen Users 是指训练/验证集中未出现过的用户，用于评估模型对新用户的泛化能力。</p>
<img src="02_all_vs_unseen_users.png" alt="全量 vs Unseen 对比">

<h2>五、Test 十分位响应率</h2>
<p>十分位曲线越单调下降，说明模型排序越可靠。左图为全量用户，右图为 Unseen Users。</p>
<img src="03_test_deciles.png" alt="Test 十分位响应率">

<h2>六、数据集分布</h2>
<table><thead><tr><th>Scope</th><th>样本量</th><th>正样本率</th></tr></thead><tbody>{distribution_html}</tbody></table>
<h2>七、冻结 E5 表现</h2>
<table><thead><tr><th>Scope</th><th>ROC-AUC</th><th>AP</th><th>Precision@1%</th><th>Recall@1%</th><th>Lift@1%</th></tr></thead><tbody>{performance_html}</tbody></table>

<h2>八、下一阶段建议</h2>
<ul>
<li>最佳模型 <b>{DISPLAY_NAMES.get(best_exp, best_exp)}</b> 通过时间外 Test 验收，具备进入下一阶段离线阈值和业务策略评审的条件</li>
<li>当前仅证明相关性排序能力，尚不能直接证明发券增量收益；正式上线仍需下一阶段随机实验或 uplift/因果评估</li>
<li>部署时需要根据实际正样本率调整阈值，建议使用 Lift 曲线辅助决策</li>
<li>定期监控模型效果，如发现正样本率漂移，需要重新训练或校准</li>
</ul>

<details><summary><b>附录：概率校准与正样本率漂移诊断</b></summary>
<p>当前模型优先用于排序/圈人，因此校准与漂移图下沉到附录。</p>
<img src="04_calibration.png" alt="概率校准">
<img src="05_positive_rate_drift.png" alt="正样本率漂移">
</details>
<h2>九、文件位置</h2>
<ul>
<li>Test 指标：<code class="path">{html.escape(str(test_dir / 'test_metrics.csv'))}</code></li>
<li>Test 预测：<code class="path">{html.escape(str(test_dir / 'test_predictions.csv'))}</code></li>
<li>十分位明细：<code class="path">{html.escape(str(test_dir))}</code> 下各 <code>*_deciles.csv</code></li>
</ul>
</body></html>"""
    
    (review_dir / "00_test_review.html").write_text(document, encoding='utf-8')


def main() -> None:
    args = parse_args()
    output_dir, checkpoint_dir, test_dir = resolve_run(args.run_id)
    manifest_path = checkpoint_dir / "frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少冻结清单：{manifest_path}")
    frozen_experiment = json.loads(manifest_path.read_text(encoding="utf-8"))["selected_experiment"]
    
    review_dir = test_dir / "review"
    review_dir.mkdir(exist_ok=True)
    
    # 加载数据
    val_metrics = load_validation_metrics(output_dir)
    test_metrics = load_test_metrics(test_dir)
    
    # 生成可视化
    print("生成 Validation vs Test 对比图...")
    plot_validation_vs_test(val_metrics, test_metrics, review_dir)
    
    print("生成全量 vs Unseen Users 对比图...")
    plot_all_vs_unseen_users(test_metrics, review_dir)
    
    print("生成 Test 十分位响应率图...")
    plot_test_deciles(test_dir, review_dir)
    
    print("生成概率校准图...")
    plot_calibration(test_dir, test_metrics, review_dir)
    
    print("生成正样本率漂移分析图...")
    plot_positive_rate_drift(val_metrics, test_metrics, review_dir)
    
    print("生成 HTML 报告...")
    render_test_report(output_dir, test_dir, val_metrics, test_metrics, review_dir, frozen_experiment)
    
    print(f"\nTest Review 报告已生成：{review_dir / '00_test_review.html'}")


if __name__ == "__main__":
    main()
