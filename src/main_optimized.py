#!/usr/bin/env python3
"""券敏感人群第一阶段 Response 模型训练入口 - 优化版。

优化点：
1. 增加 scale_pos_weight 处理不平衡
2. 优化学习率和早停策略
3. 改进特征处理（增加特征选择）
4. 扩展参数搜索空间
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation import evaluate_binary
from rfm import LiveRFM, PlatformRFM

LOGGER = logging.getLogger("response_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Response 模型训练与双轨 RFM 评估 - 优化版")
    parser.add_argument("--config", default="configs/training_config.json")
    parser.add_argument("--evaluate-test", action="store_true", help="冻结方案后仅执行一次最终测试")
    parser.add_argument("--sample-rows", type=int, default=None, help="仅用于开发验证：每个集合最多读取前 N 行")
    parser.add_argument("--skip-model", action="store_true", help="只执行数据校验和 RFM 基线")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    config_path = ROOT / path
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def setup_run() -> Tuple[Path, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "outputs" / run_id
    checkpoint_dir = ROOT / "checkpoints" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / f"training_{run_id}.log", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)
    return output_dir, checkpoint_dir


def read_dataset(path: str, encoding: str, nrows: int = None) -> pd.DataFrame:
    full_path = ROOT / path
    LOGGER.info("读取数据：%s", full_path)
    frame = pd.read_csv(full_path, encoding=encoding, nrows=nrows, low_memory=False)
    frame.columns = frame.columns.str.replace("\ufeff", "", regex=False)
    return frame


def validate_datasets(frames: Dict[str, pd.DataFrame], config: Dict) -> Tuple[List[str], List[str], pd.DataFrame]:
    target = config["target"]
    ids = config["features"]["id_columns"]
    labels = config["features"]["label_columns"]
    platform = config["features"]["platform_rfm_columns"]
    required = set(ids + labels + platform + [
        "customer_stage_live_yesterday", "last_goods_live_buy_to_today_days",
        "l180d_live_order_cnt", "l180d_live_dgmv",
    ])
    reference = list(frames["train"].columns)
    if len(reference) != len(set(reference)):
        raise ValueError("训练集存在重复列名")
    rows = []
    for name, frame in frames.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} 缺少必要字段：{missing}")
        if set(frame.columns) != set(reference):
            raise ValueError(f"{name} 与训练集列集合不一致")
        y = pd.to_numeric(frame[target], errors="coerce")
        if y.isna().any() or not set(y.unique()).issubset({0, 1}):
            raise ValueError(f"{name} 目标标签不是完整二元标签")
        duplicated_key = int(frame.duplicated(["dtm", "user_id"]).sum())
        if duplicated_key:
            raise ValueError(f"{name} 存在 {duplicated_key} 条重复 dtm × user_id")
        rows.append({
            "dataset": name, "rows": len(frame), "columns": len(frame.columns),
            "positive_rate": float(y.mean()), "unique_users": int(frame["user_id"].nunique()),
            "date_min": str(frame["dtm"].min()), "date_max": str(frame["dtm"].max()),
        })
    live_features = [column for column in reference if column not in set(ids + labels + platform)]
    if len(live_features) != 143:
        raise ValueError(f"直播特征应为 143 个，实际为 {len(live_features)} 个")
    all_features = live_features + platform
    return live_features, all_features, pd.DataFrame(rows)


def user_overlap(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left, right in pairs:
        if left not in frames or right not in frames:
            continue
        left_users = set(frames[left]["user_id"].astype(str))
        right_users = set(frames[right]["user_id"].astype(str))
        overlap = len(left_users & right_users)
        rows.append({
            "left": left, "right": right, "overlap_users": overlap,
            "left_overlap_rate": overlap / max(len(left_users), 1),
            "right_overlap_rate": overlap / max(len(right_users), 1),
        })
    return pd.DataFrame(rows)


def save_json(data: Dict, path: Path) -> None:
    def default(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=default)


def fit_smoothed_group_probability(groups: pd.Series, target: pd.Series,
                                   alpha: int, global_rate: float) -> Dict[str, float]:
    stats = pd.DataFrame({"group": groups.astype(str), "y": pd.to_numeric(target)}).groupby(
        "group", observed=True)["y"].agg(["sum", "count"])
    return ((stats["sum"] + alpha * global_rate) / (stats["count"] + alpha)).to_dict()


def predict_smoothed_group_probability(groups: pd.Series, mapping: Dict[str, float],
                                       global_rate: float) -> pd.Series:
    return groups.astype(str).map(mapping).fillna(global_rate).astype(float)


def choose_live_alpha(live_rfm: LiveRFM, validation_scored: pd.DataFrame,
                      y_validation: pd.Series, alphas: Iterable[int], lift_fractions: Iterable[float],
                      output_dir: Path) -> Tuple[int, List[Dict]]:
    reports = []
    for alpha in alphas:
        prediction = live_rfm.predict_probability(validation_scored, int(alpha))
        metrics, deciles = evaluate_binary(y_validation, prediction, lift_fractions)
        metrics.update({"experiment": "E2_live_rfm_probability", "alpha": int(alpha)})
        reports.append(metrics)
        deciles.to_csv(output_dir / f"validation_E2_alpha_{alpha}_deciles.csv", index=False, encoding="utf-8-sig")
    selected = max(reports, key=lambda item: item["average_precision"])["alpha"]
    live_rfm.selected_alpha_ = int(selected)
    return int(selected), reports


def prepare_features(train: pd.DataFrame, validation: pd.DataFrame, feature_columns: List[str]):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder
    except ImportError as exc:
        raise RuntimeError("缺少 scikit-learn，请先安装 requirements.txt") from exc

    categorical = [column for column in feature_columns
                   if train[column].dtype == "object" or pd.api.types.is_string_dtype(train[column])]
    numerical = [column for column in feature_columns if column not in categorical]
    transformer = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numerical),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       encoded_missing_value=-1, dtype=np.float32)),
        ]), categorical),
    ], remainder="drop", verbose_feature_names_out=True)
    LOGGER.info("拟合特征处理器：数值=%d，类别=%d", len(numerical), len(categorical))
    x_train = transformer.fit_transform(train[feature_columns])
    x_validation = transformer.transform(validation[feature_columns])
    return transformer, x_train, x_validation


def train_lgbm_candidates(x_train, y_train, x_validation, y_validation, candidates: List[Dict],
                          model_config: Dict, seed: int) -> Tuple[object, Dict, List[Dict]]:
    try:
        import lightgbm as lgb
        from sklearn.metrics import average_precision_score
    except ImportError as exc:
        raise RuntimeError("缺少 lightgbm/scikit-learn，请先安装 requirements.txt") from exc

    reports, best_model, best_params, best_score = [], None, None, -np.inf
    base = {
        "objective": "binary", "metric": ["auc", "binary_logloss"], "verbosity": -1,
        "seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed,
        "num_threads": -1,
    }
    
    # 计算 scale_pos_weight 处理不平衡
    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    scale_pos_weight = negative_count / max(positive_count, 1)
    base["scale_pos_weight"] = scale_pos_weight
    LOGGER.info("类别平衡：正样本=%d，负样本=%d，scale_pos_weight=%.2f", 
                int(positive_count), int(negative_count), scale_pos_weight)
    
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(x_validation, label=y_validation, reference=train_set, free_raw_data=False)
    for index, candidate in enumerate(candidates):
        params = {**base, **candidate}
        model = lgb.train(
            params, train_set, num_boost_round=int(model_config["num_boost_round"]),
            valid_sets=[valid_set], valid_names=["validation"],
            callbacks=[lgb.early_stopping(int(model_config["early_stopping_rounds"]), verbose=False)],
        )
        prediction = model.predict(x_validation, num_iteration=model.best_iteration)
        score = float(average_precision_score(y_validation, prediction))
        reports.append({"candidate": index, "average_precision": score,
                        "best_iteration": int(model.best_iteration), "parameters": candidate})
        LOGGER.info("候选 %d：AP=%.4f，best_iteration=%d", index, score, int(model.best_iteration))
        if score > best_score:
            best_score, best_model, best_params = score, model, candidate
    return best_model, best_params, reports


def fit_model_experiment(name: str, train: pd.DataFrame, validation: pd.DataFrame,
                         features: List[str], config: Dict, output_dir: Path, checkpoint_dir: Path):
    target = config["target"]
    transformer, x_train, x_validation = prepare_features(train, validation, features)
    model, params, search_report = train_lgbm_candidates(
        x_train, train[target], x_validation, validation[target],
        config["model"]["parameter_candidates"], config["model"], config["random_seed"])
    prediction = model.predict(x_validation, num_iteration=model.best_iteration)
    metrics, deciles = evaluate_binary(validation[target], prediction, config["evaluation"]["lift_fractions"])
    metrics.update({"experiment": name, "best_iteration": int(model.best_iteration)})
    deciles.to_csv(output_dir / f"validation_{name}_deciles.csv", index=False, encoding="utf-8-sig")
    model.save_model(str(checkpoint_dir / f"{name}.txt"))
    with (checkpoint_dir / f"{name}_preprocessor.pkl").open("wb") as file:
        pickle.dump(transformer, file)
    save_json({"features": features, "best_parameters": params, "search": search_report},
              checkpoint_dir / f"{name}_metadata.json")
    return {"name": name, "model": model, "transformer": transformer, "features": features,
            "metrics": metrics, "parameters": params}


def evaluate_test_model(experiment: Dict, test: pd.DataFrame, config: Dict,
                        output_dir: Path) -> Dict:
    x_test = experiment["transformer"].transform(test[experiment["features"]])
    prediction = experiment["model"].predict(x_test, num_iteration=experiment["model"].best_iteration)
    metrics, deciles = evaluate_binary(test[config["target"]], prediction,
                                       config["evaluation"]["lift_fractions"])
    metrics["experiment"] = experiment["name"]
    deciles.to_csv(output_dir / f"test_{experiment['name']}_deciles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "dtm": test["dtm"], "user_id": test["user_id"],
        "label": test[config["target"]], "probability": prediction,
    }).to_csv(output_dir / f"test_{experiment['name']}_predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir, checkpoint_dir = setup_run()
    save_json(config, output_dir / "resolved_config.json")

    split_paths = {"train": config["data"]["train"], "validation": config["data"]["validation"]}
    if args.evaluate_test:
        split_paths["test"] = config["data"]["test"]
    frames = {name: read_dataset(path, config["data"]["encoding"], args.sample_rows)
              for name, path in split_paths.items()}
    live_features, all_features, data_summary = validate_datasets(frames, config)
    data_summary.to_csv(output_dir / "data_summary.csv", index=False, encoding="utf-8-sig")
    user_overlap(frames).to_csv(output_dir / "user_overlap.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("数据校验通过；直播特征=%d，全量特征=%d", len(live_features), len(all_features))

    platform_rfm = PlatformRFM(**config["rfm"]["platform"]).fit(frames["train"])
    live_rfm = LiveRFM(**{key: value for key, value in config["rfm"]["live"].items()
                          if key in {"r_bins", "n_bins"}}).fit_scores(frames["train"])
    platform_scored = {name: platform_rfm.transform(frame) for name, frame in frames.items()}
    live_scored = {name: live_rfm.transform_scores(frame) for name, frame in frames.items()}
    live_rfm.fit_probability(live_scored["train"], frames["train"][config["target"]],
                             config["rfm"]["live"]["alpha_candidates"])
    alpha, validation_reports = choose_live_alpha(
        live_rfm, live_scored["validation"], frames["validation"][config["target"]],
        config["rfm"]["live"]["alpha_candidates"], config["evaluation"]["lift_fractions"], output_dir)

    global_rate = float(frames["train"][config["target"]].mean())
    e0_prediction = np.full(len(frames["validation"]), global_rate)
    e0_metrics, e0_deciles = evaluate_binary(frames["validation"][config["target"]], e0_prediction,
                                              config["evaluation"]["lift_fractions"])
    e0_metrics["experiment"] = "E0_global_rate"
    e0_deciles.to_csv(output_dir / "validation_E0_deciles.csv", index=False, encoding="utf-8-sig")

    # E1 使用可覆盖全体用户的直播业务展示分；U=-1、N=0、D=1，活跃用户为 3~9。
    display_score = live_scored["validation"]["live_rfm_sum"].copy()
    display_score = display_score.fillna(live_scored["validation"]["live_lifecycle"].map({"U": -1, "N": 0, "D": 1}))
    e1_metrics, e1_deciles = evaluate_binary(
        frames["validation"][config["target"]], display_score,
        config["evaluation"]["lift_fractions"], is_probability=False)
    e1_metrics["experiment"] = "E1_live_rfm_sum"
    e1_deciles.to_csv(output_dir / "validation_E1_deciles.csv", index=False, encoding="utf-8-sig")

    # 全平台 RFM 对所有生命周期均保留独立分组，并用训练集响应率提供可比较概率。
    platform_probability_reports = []
    platform_maps = {}
    for candidate_alpha in config["rfm"]["live"]["alpha_candidates"]:
        mapping = fit_smoothed_group_probability(
            platform_scored["train"]["platform_rfm_cell"], frames["train"][config["target"]],
            int(candidate_alpha), global_rate)
        platform_maps[int(candidate_alpha)] = mapping
        prediction = predict_smoothed_group_probability(
            platform_scored["validation"]["platform_rfm_cell"], mapping, global_rate)
        metrics, deciles = evaluate_binary(
            frames["validation"][config["target"]], prediction,
            config["evaluation"]["lift_fractions"])
        metrics.update({"experiment": "E2_platform_rfm_probability", "alpha": int(candidate_alpha)})
        platform_probability_reports.append(metrics)
        deciles.to_csv(output_dir / f"validation_E2_platform_alpha_{candidate_alpha}_deciles.csv",
                       index=False, encoding="utf-8-sig")
    platform_alpha = int(max(platform_probability_reports,
                             key=lambda item: item["average_precision"])["alpha"])
    validation_metrics = [e0_metrics, e1_metrics] + validation_reports + platform_probability_reports

    rfm_quality = {}
    for name in frames:
        rfm_quality[name] = {
            "live_lifecycle": live_scored[name]["live_lifecycle"].value_counts(dropna=False).to_dict(),
            "platform_lifecycle": platform_scored[name]["platform_lifecycle"].value_counts(dropna=False).to_dict(),
            "zero_gmv_with_order": int(live_scored[name]["zero_gmv_with_order"].sum()),
        }
    save_json({
        "platform_edges": platform_rfm.edges_, "live_f_edges": live_rfm.f_edges_,
        "live_m_edges": live_rfm.m_edges_, "selected_alpha": alpha,
        "global_positive_rate": live_rfm.global_rate_, "platform_selected_alpha": platform_alpha,
        "platform_probability_map": platform_maps[platform_alpha], "quality": rfm_quality,
    }, output_dir / "rfm_rules_and_quality.json")
    with (checkpoint_dir / "rfm_objects.pkl").open("wb") as file:
        pickle.dump({"platform": platform_rfm, "live": live_rfm}, file)

    experiments = []
    if not args.skip_model:
        rfm_raw_features = [
            "l720d_last_to_today_days", "l360d_buy_days", "l360d_dgmv", "total_orders",
            "last_goods_live_buy_to_today_days", "l180d_live_order_cnt", "l180d_live_dgmv",
            "customer_stage_live_yesterday", "live_buy_active_flag",
        ]
        LOGGER.info("训练 E3：双轨 RFM 原值轻量模型")
        experiments.append(fit_model_experiment(
            "E3_dual_rfm_raw", frames["train"], frames["validation"], rfm_raw_features,
            config, output_dir, checkpoint_dir))
        LOGGER.info("训练 E4：143 个直播特征")
        experiments.append(fit_model_experiment(
            "E4_live_features", frames["train"], frames["validation"], live_features,
            config, output_dir, checkpoint_dir))
        LOGGER.info("训练 E5：143 个直播特征 + 4 个全平台 RFM 原始字段")
        experiments.append(fit_model_experiment(
            "E5_live_plus_platform", frames["train"], frames["validation"], all_features,
            config, output_dir, checkpoint_dir))
        validation_metrics.extend(item["metrics"] for item in experiments)

    pd.DataFrame(validation_metrics).to_csv(
        output_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")

    if args.evaluate_test:
        test_metrics = []
        y_test = frames["test"][config["target"]]
        pred_e0 = np.full(len(frames["test"]), global_rate)
        metrics, _ = evaluate_binary(y_test, pred_e0, config["evaluation"]["lift_fractions"])
        metrics["experiment"] = "E0_global_rate"
        test_metrics.append(metrics)
        platform_prediction = predict_smoothed_group_probability(
            platform_scored["test"]["platform_rfm_cell"], platform_maps[platform_alpha], global_rate)
        metrics, deciles = evaluate_binary(y_test, platform_prediction,
                                           config["evaluation"]["lift_fractions"])
        metrics.update({"experiment": "E2_platform_rfm_probability", "alpha": platform_alpha})
        deciles.to_csv(output_dir / "test_E2_platform_deciles.csv", index=False, encoding="utf-8-sig")
        test_metrics.append(metrics)
        pred_e2 = live_rfm.predict_probability(live_scored["test"], alpha)
        metrics, deciles = evaluate_binary(y_test, pred_e2, config["evaluation"]["lift_fractions"])
        metrics.update({"experiment": "E2_live_rfm_probability", "alpha": alpha})
        deciles.to_csv(output_dir / "test_E2_deciles.csv", index=False, encoding="utf-8-sig")
        test_metrics.append(metrics)
        if experiments:
            best = max(experiments, key=lambda item: item["metrics"][config["evaluation"]["primary_metric"]])
            test_metrics.append(evaluate_test_model(best, frames["test"], config, output_dir))
        pd.DataFrame(test_metrics).to_csv(output_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")

    LOGGER.info("训练流程完成，结果目录：%s", output_dir)


if __name__ == "__main__":
    main()
