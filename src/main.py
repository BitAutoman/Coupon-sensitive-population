#!/usr/bin/env python3
"""券敏感人群第一阶段 Response 模型训练入口。

流程：数据校验 -> 双轨 RFM -> E0/E1/E2 基线 -> E4/E5 LightGBM 选模 -> 冻结后测试。
默认不会读取测试集，只有显式传入 --evaluate-test 才执行最终测试评估。
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
    parser = argparse.ArgumentParser(description="Response 模型训练与双轨 RFM 评估")
    parser.add_argument("--config", default="configs/training_config.json")
    parser.add_argument("--evaluate-test", action="store_true",
                       help="从已冻结 checkpoint 加载模型并执行一次最终测试；必须同时指定 --run-id")
    parser.add_argument("--run-id", type=str, default=None,
                       help="已冻结训练运行 ID，例如 20260825_162323；测试阶段不会重新训练或选参")
    parser.add_argument("--dry-run", action="store_true",
                       help="只校验冻结产物能否加载，不读取测试集、不产生测试指标")
    parser.add_argument("--sample-rows", type=int, default=None, help="仅用于开发验证：每个集合最多读取前 N 行")
    parser.add_argument("--skip-model", action="store_true", help="只执行数据校验和 RFM 基线")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    config_path = ROOT / path
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def hash_user_ids(values: pd.Series) -> np.ndarray:
    """保存不可逆的稳定 64 位用户摘要，用于测试集未见用户评估。"""
    hashes = pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(dtype=np.uint64)
    return np.unique(hashes)


def freeze_training_selection(output_dir: Path, checkpoint_dir: Path,
                              config: Dict, validation_metrics: List[Dict],
                              frames: Dict[str, pd.DataFrame]) -> Dict:
    """将验证集选择结果固化为测试阶段唯一可消费的清单。"""
    model_names = {"E3_dual_rfm_raw", "E4_live_features", "E5_live_plus_platform"}
    candidates = [item for item in validation_metrics if item.get("experiment") in model_names]
    if not candidates:
        raise RuntimeError("没有可冻结的模型实验")
    primary_metric = config["evaluation"]["primary_metric"]
    selected = max(candidates, key=lambda item: item[primary_metric])
    experiment = selected["experiment"]
    manifest = {
        "run_id": output_dir.name,
        "status": "frozen_for_test",
        "selection_source": "validation_only",
        "primary_metric": primary_metric,
        "selected_experiment": experiment,
        "validation_metric": float(selected[primary_metric]),
        "best_iteration": int(selected["best_iteration"]),
        "model_file": f"{experiment}.txt",
        "preprocessor_file": f"{experiment}_preprocessor.pkl",
        "metadata_file": f"{experiment}_metadata.json",
        "rfm_file": "rfm_objects.pkl",
        "resolved_config_file": str(output_dir / "resolved_config.json"),
    }
    save_json(manifest, checkpoint_dir / "frozen_manifest.json")
    seen_hashes = np.unique(np.concatenate([
        hash_user_ids(frames["train"]["user_id"]),
        hash_user_ids(frames["validation"]["user_id"]),
    ]))
    np.save(checkpoint_dir / "seen_user_hashes.npy", seen_hashes, allow_pickle=False)
    return manifest


def setup_test_logging(run_id: str) -> None:
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(
            ROOT / "logs" / f"test_{run_id}.log", encoding="utf-8")],
        force=True,
    )


def load_frozen_bundle(run_id: str) -> Tuple[Dict, Dict, object, object, Dict, np.ndarray]:
    """加载验证集阶段冻结的唯一模型及配套预处理/RFM，不做任何拟合。"""
    checkpoint_dir = ROOT / "checkpoints" / run_id
    output_dir = ROOT / "outputs" / run_id
    manifest_path = checkpoint_dir / "frozen_manifest.json"
    required = [manifest_path, output_dir / "resolved_config.json",
                checkpoint_dir / "rfm_objects.pkl", checkpoint_dir / "seen_user_hashes.npy"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"checkpoint 尚未完整冻结，缺少：{missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((checkpoint_dir / manifest["metadata_file"]).read_text(encoding="utf-8"))
    with (checkpoint_dir / manifest["preprocessor_file"]).open("rb") as file:
        preprocessor = pickle.load(file)
    with (checkpoint_dir / manifest["rfm_file"]).open("rb") as file:
        rfm_objects = pickle.load(file)
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("缺少 lightgbm，请先安装 requirements.txt") from exc
    model = lgb.Booster(model_file=str(checkpoint_dir / manifest["model_file"]))
    seen_hashes = np.load(checkpoint_dir / "seen_user_hashes.npy", allow_pickle=False)
    return manifest, config, model, preprocessor, {**rfm_objects, "metadata": metadata}, seen_hashes


def run_frozen_test(args: argparse.Namespace) -> None:
    if not args.run_id:
        raise ValueError("--evaluate-test 必须同时指定 --run-id，例如：--run-id 20260825_162323")
    setup_test_logging(args.run_id)
    manifest, config, model, preprocessor, bundle, seen_hashes = load_frozen_bundle(args.run_id)
    LOGGER.info("冻结产物加载成功：run_id=%s，模型=%s，validation_%s=%.6f",
                args.run_id, manifest["selected_experiment"], manifest["primary_metric"],
                manifest["validation_metric"])
    if args.dry_run:
        LOGGER.info("dry-run 完成：未读取测试集、未生成预测、未写测试指标")
        return
    if args.sample_rows is not None:
        raise ValueError("正式 checkpoint 测试禁止使用 --sample-rows；如需校验加载请使用 --dry-run")

    test_dir = ROOT / "outputs" / args.run_id / "test"
    if (test_dir / "test_metrics.csv").exists():
        raise FileExistsError(f"该 checkpoint 已有正式测试结果，拒绝重复评估：{test_dir}")
    test_dir.mkdir(parents=True, exist_ok=True)
    test = read_dataset(config["data"]["test"], config["data"]["encoding"])
    target = config["target"]
    features = bundle["metadata"]["features"]
    required = set(config["features"]["id_columns"] + [target] + features)
    missing = sorted(required - set(test.columns))
    if missing:
        raise ValueError(f"测试集缺少冻结模型所需字段：{missing}")
    y_test = pd.to_numeric(test[target], errors="coerce").reset_index(drop=True)
    if y_test.isna().any() or not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("测试集目标标签不是完整二元标签")
    test_hashes = pd.util.hash_pandas_object(test["user_id"].astype(str), index=False).to_numpy(dtype=np.uint64)
    unseen_mask = ~np.isin(test_hashes, seen_hashes)

    rules = json.loads((ROOT / "outputs" / args.run_id / "rfm_rules_and_quality.json").read_text(encoding="utf-8"))
    global_rate = float(rules["global_positive_rate"])
    platform_scored = bundle["platform"].transform(test)
    live_scored = bundle["live"].transform_scores(test)
    predictions = {
        "E0_global_rate": np.full(len(test), global_rate),
        "E2_platform_rfm_probability": predict_smoothed_group_probability(
            platform_scored["platform_rfm_cell"], rules["platform_probability_map"], global_rate).to_numpy(),
        "E2_live_rfm_probability": bundle["live"].predict_probability(
            live_scored, int(bundle["live"].selected_alpha_)).to_numpy(),
    }
    transformed = preprocessor.transform(test[features])
    predictions[manifest["selected_experiment"]] = model.predict(
        transformed, num_iteration=int(manifest["best_iteration"]))

    reports = []
    for experiment, prediction in predictions.items():
        scope_reports, scope_deciles = evaluate_scopes(
            y_test, prediction, unseen_mask, config["evaluation"]["lift_fractions"], experiment)
        reports.extend(scope_reports)
        for scope, deciles in scope_deciles.items():
            deciles.to_csv(test_dir / f"{experiment}_{scope}_deciles.csv",
                           index=False, encoding="utf-8-sig")
    pd.DataFrame(reports).to_csv(test_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_frame = pd.DataFrame({
        "dtm": test["dtm"], "user_id": test["user_id"], "label": y_test,
        "is_unseen_user": unseen_mask,
    })
    for experiment, prediction in predictions.items():
        prediction_frame[experiment] = prediction
    prediction_frame.to_csv(test_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    save_json({"run_id": args.run_id, "manifest": manifest, "test_rows": len(test),
               "unseen_rows": int(unseen_mask.sum())}, test_dir / "test_execution.json")
    LOGGER.info("冻结模型测试完成；未执行训练或选参。结果目录：%s", test_dir)


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
        # 同一 Dataset 上搜索不同 min_child_samples 时必须关闭预过滤，
        # 否则后续候选降低叶节点最小样本数会触发 LightGBM 警告并可能遗漏特征。
        "feature_pre_filter": False,
    }
    
    # 处理类别不平衡
    if model_config.get("is_unbalance", False):
        base["is_unbalance"] = True
        LOGGER.info("启用 is_unbalance 处理类别不平衡")
    
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(x_validation, label=y_validation, reference=train_set, free_raw_data=False)
    for index, candidate in enumerate(candidates):
        params = {**base, **candidate}
        LOGGER.info("训练候选模型 %d/%d", index + 1, len(candidates))
        model = lgb.train(
            params, train_set, num_boost_round=int(model_config["num_boost_round"]),
            valid_sets=[valid_set], valid_names=["validation"],
            callbacks=[lgb.early_stopping(int(model_config["early_stopping_rounds"]), verbose=False)],
        )
        prediction = model.predict(x_validation, num_iteration=model.best_iteration)
        score = float(average_precision_score(y_validation, prediction))
        reports.append({"candidate": index, "average_precision": score,
                        "best_iteration": int(model.best_iteration), "parameters": candidate})
        LOGGER.info("候选 %d: AP=%.4f, best_iteration=%d", index, score, int(model.best_iteration))
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
    
    # 必须使用预处理器的真实输出列名；ColumnTransformer 会重排数值/类别列，
    # SimpleImputer 还可能追加缺失指示器，不能直接与原始 features 按位置配对。
    transformed_names = transformer.get_feature_names_out().tolist()
    importance = model.feature_importance(importance_type="gain")
    if len(transformed_names) != len(importance):
        raise RuntimeError(
            f"变换后特征名数量 {len(transformed_names)} 与模型重要性数量 {len(importance)} 不一致")
    feature_importance = sorted(zip(transformed_names, importance), key=lambda item: item[1], reverse=True)
    total_gain = max(float(np.sum(importance)), 1.0)
    top_features = [
        {"feature": feature, "importance": float(gain), "gain_ratio": float(gain / total_gain)}
        for feature, gain in feature_importance[:20]
    ]
    
    save_json({
        "features": features,
        "best_parameters": params,
        "search": search_report,
        "top_features": top_features
    }, checkpoint_dir / f"{name}_metadata.json")
    
    # 保存完整特征重要性
    importance_frame = pd.DataFrame(feature_importance, columns=["feature", "importance"])
    importance_frame["gain_ratio"] = importance_frame["importance"] / total_gain
    importance_frame.to_csv(
        output_dir / f"{name}_feature_importance.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("Top 5 特征: %s", [f[0] for f in feature_importance[:5]])
    
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


def evaluate_scopes(y_true: pd.Series, prediction: np.ndarray, unseen_mask: pd.Series,
                    lift_fractions: Iterable[float], experiment: str) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
    """同时评估测试全量用户和相对 Train/Validation 均未出现的用户。"""
    reports, deciles = [], {}
    masks = {"all_users": np.ones(len(y_true), dtype=bool), "unseen_users": np.asarray(unseen_mask, dtype=bool)}
    for scope, mask in masks.items():
        if int(mask.sum()) == 0:
            LOGGER.warning("%s 没有样本，跳过 %s", scope, experiment)
            continue
        metrics, scope_deciles = evaluate_binary(
            y_true.iloc[np.flatnonzero(mask)], np.asarray(prediction)[mask], lift_fractions)
        metrics.update({"experiment": experiment, "evaluation_scope": scope})
        reports.append(metrics)
        deciles[scope] = scope_deciles
    return reports, deciles


def main() -> None:
    args = parse_args()
    if args.dry_run and not args.evaluate_test:
        raise ValueError("--dry-run 只能与 --evaluate-test --run-id 一起使用")
    if args.evaluate_test:
        run_frozen_test(args)
        return
    if args.run_id:
        raise ValueError("--run-id 仅用于 --evaluate-test，训练阶段会自动生成新的 run-id")
    config = load_config(args.config)
    output_dir, checkpoint_dir = setup_run()
    save_json(config, output_dir / "resolved_config.json")

    split_paths = {"train": config["data"]["train"], "validation": config["data"]["validation"]}
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
    rfm_segment_rows = []
    for name in frames:
        rfm_quality[name] = {
            "live_lifecycle": live_scored[name]["live_lifecycle"].value_counts(dropna=False).to_dict(),
            "platform_lifecycle": platform_scored[name]["platform_lifecycle"].value_counts(dropna=False).to_dict(),
            "zero_gmv_with_order": int(live_scored[name]["zero_gmv_with_order"].sum()),
        }
        target_values = pd.to_numeric(frames[name][config["target"]], errors="coerce")
        for rfm_type, lifecycle in [
            ("live", live_scored[name]["live_lifecycle"]),
            ("platform", platform_scored[name]["platform_lifecycle"]),
        ]:
            segment_frame = pd.DataFrame({"segment": lifecycle.astype(str), "target": target_values})
            summary = segment_frame.groupby("segment", observed=True)["target"].agg(
                sample_count="size", positive_count="sum", response_rate="mean").reset_index()
            summary["sample_share"] = summary["sample_count"] / len(segment_frame)
            summary.insert(0, "dataset", name)
            summary.insert(0, "rfm_type", rfm_type)
            rfm_segment_rows.append(summary)
    pd.concat(rfm_segment_rows, ignore_index=True).to_csv(
        output_dir / "rfm_segment_summary.csv", index=False, encoding="utf-8-sig")
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

    if experiments:
        manifest = freeze_training_selection(
            output_dir, checkpoint_dir, config, validation_metrics, frames)
        LOGGER.info("已冻结最佳模型：%s，%s=%.4f；后续测试必须指定 --run-id %s",
                    manifest["selected_experiment"], manifest["primary_metric"],
                    manifest["validation_metric"], manifest["run_id"])

    LOGGER.info("训练流程完成，结果目录：%s", output_dir)


if __name__ == "__main__":
    main()
