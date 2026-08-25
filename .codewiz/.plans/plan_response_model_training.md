# Response 模型训练实现计划

## 概述

本计划旨在实现券敏感人群第一阶段的 Response 模型训练，核心任务包括：
1. 设计双轨 RFM 打分规则（全平台 RFM + 直播 RFM）
2. 编写完整的模型训练代码，目标标签为 `flag_repurchase_live_paid_14d`

## 架构分析

### 现有资源
- **数据**：训练集（867K 行）、验证集（167K 行）、测试集（161K 行）已就位
- **特征**：143 个直播特征 + 4 个全平台 RFM 字段（已在 CSV 中，无需额外补取）
- **文档**：完整的建模方案和 RFM 打分规则文档
- **历史项目**：`/Users/yukewei/Public/xhs/RFM` 提供全平台和直播间 RFM 参考实现

### 关键技术决策
1. **双轨 RFM 设计**：
   - 全平台 RFM：沿用历史项目思想（五分位打分 + 10 类分层），但修正 `rank(method="first")` 的不稳健性
   - 直播 RFM：按 `docs/直播RFM优化后完整打分规则.md` 实现（三档打分 + A/D/N/U 生命周期预分层 + 训练集平滑响应率）

2. **模型训练流程**：
   - 数据校验 → 特征配置 → RFM 基线 → LightGBM 主模型 → 验证集选模 → 测试集一次评估

3. **评估指标**：
   - 主指标：ROC-AUC、PR-AUC、Lift@K（1%, 5%, 10%, 20%）
   - 辅助指标：LogLoss、Brier Score、Top-K Precision/Recall、十分位实际正样本率

## 实现步骤

### 阶段 1：项目结构搭建与配置管理

#### 1.1 创建项目目录结构
```
src/
├── __init__.py
├── main.py                    # 主入口
├── config.py                  # 配置管理
├── data/
│   ├── __init__.py
│   ├── loader.py              # 数据加载与校验
│   └── validator.py           # 数据质量检查
├── features/
│   ├── __init__.py
│   ├── rfm_platform.py        # 全平台 RFM 打分
│   ├── rfm_live.py            # 直播 RFM 打分
│   └── feature_config.py      # 特征配置与黑名单
├── models/
│   ├── __init__.py
│   ├── baseline.py            # RFM 基线模型
│   ├── lgbm_trainer.py        # LightGBM 训练器
│   └── evaluator.py           # 统一评估器
└── utils/
    ├── __init__.py
    ├── logger.py              # 日志工具
    └── metrics.py             # 指标计算
```

#### 1.2 配置文件设计
创建 `configs/config.yaml`：
```yaml
# 数据路径
data:
  train: data/raw/训练集数据.csv
  validation: data/raw/验证集数据.csv
  test: data/raw/测试集数据.csv

# 目标标签
target: flag_repurchase_live_paid_14d

# 特征配置
features:
  # 主键/标识（不输入模型）
  id_cols:
    - user_id
    - dtm
    - user_type
  
  # Label 黑名单（9 个 Label 除目标外其余 8 个必须剔除）
  label_blacklist:
    - flag_repurchase_14d
    - flag_repurchase_paid_14d
    - flag_repurchase_live_14d
    - flag_repurchase_new_category_14d
    - flag_repurchase_new_category_paid_14d
    - repurchase_days_14d
    - repurchase_pgmv_14d
    - repurchase_dgmv_14d
  
  # 全平台 RFM 字段（可选，用于 E5 实验）
  platform_rfm_cols:
    - l720d_last_to_today_days
    - l360d_buy_days
    - l360d_dgmv
    - total_orders

# 全平台 RFM 配置
platform_rfm:
  maturity_threshold: 3  # total_orders >= 3 才进入打分
  n_bins: 5  # 五分位打分
  
# 直播 RFM 配置
live_rfm:
  rule_version: live_rfm_response_v1
  alpha: 100  # 平滑强度初始值
  alpha_candidates: [20, 50, 100, 200]  # 验证集选择
  
# LightGBM 配置
lgbm:
  params:
    objective: binary
    metric: [binary_logloss, auc]
    boosting_type: gbdt
    num_leaves: 63
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    verbose: -1
  early_stopping_rounds: 100
  num_boost_round: 1000

# 实验矩阵
experiments:
  - name: E0
    desc: 全局正样本率基线
    features: []
  - name: E1
    desc: 直播 RFM_sum
    features: [rfm_sum]
  - name: E2
    desc: 直播 RFM 平滑响应率
    features: [rfm_probability]
  - name: E3
    desc: 直播 RFM 原值 + 逻辑回归/LightGBM
    features: [r_raw, f_raw, m_raw]
  - name: E4
    desc: 143 个直播特征 LightGBM（主模型）
    features: all_live
  - name: E5
    desc: 143 直播特征 + 4 全平台 RFM 字段
    features: all_live + platform_rfm
  - name: E6
    desc: E4 + RFM 派生特征
    features: all_live + rfm_derived

# 输出路径
output:
  checkpoints: checkpoints/
  logs: logs/
  outputs: outputs/
```

### 阶段 2：数据加载与校验模块

#### 2.1 数据加载器 (`src/data/loader.py`)
- 统一使用 `encoding='utf-8-sig'` 处理 BOM 差异
- 自动识别并分离主键、Label、特征
- 应用 Label 黑名单过滤
- 数据类型自动推断与转换

#### 2.2 数据校验器 (`src/data/validator.py`)
**必须检查项**：
1. **样本量统计**：三个集合各自的样本量
2. **正样本率**：目标标签在三个集合的正样本率
3. **用户重合度**：跨集合重复 user_id 数量与比例
4. **空值检查**：143 个特征的空值比例
5. **特征时间窗口**：确认所有特征严格截至 dtm 前（无穿越）
6. **标签定义**：确认 `flag_repurchase_live_paid_14d` 的精确定义

**输出报告**：
```
data_validation_report.txt
- 训练集：867,476 样本，正样本率 6.07%
- 验证集：167,556 样本，正样本率 7.18%
- 测试集：161,719 样本，正样本率 9.06%
- 跨集合用户重合：Train-Val 重叠 X%，Train-Test 重叠 Y%
- 空值特征：列出空值率 > 5% 的特征
```

### 阶段 3：双轨 RFM 打分模块

#### 3.1 全平台 RFM (`src/features/rfm_platform.py`)

**字段映射**：
- R = `l720d_last_to_today_days`（末单距今天数）
- F = `l360d_buy_days`（近 360 天购买天数）
- M = `l360d_dgmv`（近 360 天 DGMV）
- 生命周期 = `total_orders`

**打分规则**（修正历史项目不稳健之处）：
1. **生命周期预分层**：仅 `total_orders >= 3` 的成熟客户进入 RFM 打分
2. **五分位打分**：
   - R：越小越好，[5, 4, 3, 2, 1]
   - F：越大越好，[1, 2, 3, 4, 5]
   - M：越大越好，[1, 2, 3, 4, 5]
3. **修正点**：
   - 不使用 `rank(method="first")`，改用 `rank(method="average")` + `qcut(duplicates="drop")`
   - 如果分位点相同，减少档数或使用固定业务阈值
   - 处理边界值：R 缺失填 721，F/M 缺失填 0

**10 类分层规则**（与历史项目一致）：
```python
Champions: R>=4, F>=4, M>=4
Promising: R>=4, F<=2, M>=3
Loyal Accounts: R>=3, F>=3, M>=3
Potential Loyalist: R>=3, F in [2,3], M in [2,3,4]
New Active Accounts: R=5, F=1
Low Spenders: R>=3, M<=2
Need Attention: R in [2,3], F<=2, M>=4
About to Sleep: R in [2,3], F<=2, M<=3
At Risk: R<=2, M>=3
Lost: R<=2, M<=2
```

**输出字段**：
```
user_id, dtm, platform_r, platform_f, platform_m, 
platform_rfm_score, platform_segment
```

#### 3.2 直播 RFM (`src/features/rfm_live.py`)

**字段映射**：
- R = `last_goods_live_buy_to_today_days`
- F = `l180d_live_order_cnt`
- M = `l180d_live_dgmv`
- 生命周期 = `customer_stage_live_yesterday`
- 辅助核对 = `live_buy_active_flag`

**第一步：生命周期预分层**
```
A（活跃）：F > 0
D（沉睡）：F = 0 且 customer_stage_live_yesterday = "直播180天流失"
N（无历史）：F = 0 且 customer_stage_live_yesterday = "直播潜客"
U（未知）：其他情况（字段冲突、缺失等）
```

**第二步：活跃用户 RFM 打分**（仅 A 类）
- R：固定阈值
  - 0~7 天 → 3
  - 8~30 天 → 2
  - 31~180 天 → 1
  - R > 180 或缺失 → 标记为 U
- F：训练集三等分位（仅对 F > 0）
  - 计算 f_cut1, f_cut2（33.33%, 66.67% 分位数）
  - 1 ≤ F ≤ f_cut1 → 1
  - f_cut1 < F ≤ f_cut2 → 2
  - F > f_cut2 → 3
- M：训练集三等分位（仅对 M > 0）
  - 计算 m_cut1, m_cut2
  - M ≤ m_cut1 → 1
  - m_cut1 < M ≤ m_cut2 → 2
  - M > m_cut2 → 3
  - F > 0 但 M = 0 → M_score = 1，标记 `zero_gmv_with_order = 1`

**第三步：业务规则总分**
```
RFM_sum = R_score + F_score + M_score  # 范围 3~9
RFM_cell = "A_R{R}F{F}M{M}"  # 精细分群编码
```

**第四步：训练集平滑响应率**
```
rfm_probability(g) = (positive_g + alpha × global_rate) / (sample_g + alpha)
```
- alpha 初始值 = 100
- 验证集从 [20, 50, 100, 200] 中选择最优 alpha
- 未见过的分组回退到同生命周期大类正样本率，再回退到全局正样本率

**输出字段**：
```
user_id, dtm, lifecycle_segment, r_raw, f_raw, m_raw,
r_score, f_score, m_score, rfm_sum, rfm_cell, rfm_probability,
rule_version
```

### 阶段 4：特征配置与工程

#### 4.1 特征配置 (`src/features/feature_config.py`)
- 定义 143 个直播特征名单
- 定义 Label 黑名单（8 个非目标 Label）
- 定义全平台 RFM 字段（4 个）
- 定义 RFM 派生特征（可选）

#### 4.2 特征工程
**基础特征**：
- 143 个直播历史特征（直接输入）
- 4 个全平台 RFM 字段（可选，用于 E5）

**RFM 派生特征**（可选，用于 E6）：
- 全平台 RFM 分数、分群
- 直播 RFM 分数、分群、响应率
- 交叉特征：直播 RFM × 全平台 RFM

### 阶段 5：模型训练与评估

#### 5.1 基线模型 (`src/models/baseline.py`)
**E0：全局正样本率**
- 预测值 = 训练集正样本率

**E1：RFM_sum**
- 使用直播 RFM_sum 作为预测分
- 非活跃用户（D/N/U）使用业务展示分（D=1, N=0, U=-1）

**E2：RFM 平滑响应率**
- 使用 rfm_probability 作为预测分

**E3：RFM 原值 + 轻量模型**
- 输入：r_raw, f_raw, m_raw
- 模型：逻辑回归或轻量 LightGBM

#### 5.2 LightGBM 主模型 (`src/models/lgbm_trainer.py`)
**训练流程**：
1. 准备训练数据（Train/Val/Test）
2. 应用特征配置（过滤黑名单、选择特征）
3. 训练 LightGBM（使用验证集早停）
4. 保存模型到 `checkpoints/`

**超参数搜索**（可选）：
- 使用 Optuna 进行贝叶斯优化
- 搜索空间：num_leaves, learning_rate, feature_fraction, bagging_fraction

#### 5.3 统一评估器 (`src/models/evaluator.py`)
**评估指标**：
```python
- ROC-AUC
- PR-AUC
- LogLoss
- Brier Score
- Lift@1%, 5%, 10%, 20%
- Top-K Precision, Recall（K=1000, 5000, 10000）
- 十分位实际正样本率
```

**评估维度**：
- 全量用户
- 跨集合未重复用户（排除 Train 中出现过的 user_id）

**输出报告**：
```
outputs/experiment_results.csv
outputs/evaluation_report.txt
outputs/decile_analysis.csv
```

### 阶段 6：实验执行与结果落盘

#### 6.1 实验矩阵执行
按顺序执行 E0~E6（E5/E6 可选）：
1. E0：全局基线
2. E1：RFM_sum
3. E2：RFM 响应率
4. E3：RFM 原值 + 轻量模型
5. E4：143 特征 LightGBM（主模型）
6. E5（可选）：143 + 4 全平台 RFM
7. E6（可选）：E4 + RFM 派生特征

#### 6.2 验证集选模
- 在验证集上比较所有实验
- 选择最优模型和超参数
- 记录选择理由

#### 6.3 测试集一次评估
- 使用最终冻结的模型在测试集上评估
- 不修改任何规则或参数
- 输出最终指标报告

#### 6.4 结果落盘
**输出文件**：
```
outputs/
├── experiment_results.csv          # 所有实验的指标对比
├── evaluation_report.txt           # 评估报告
├── decile_analysis.csv             # 十分位分析
├── model_predictions/
│   ├── train_predictions.csv       # 训练集预测
│   ├── val_predictions.csv         # 验证集预测
│   └── test_predictions.csv        # 测试集预测
└── feature_importance.csv          # 特征重要性
```

## 关键文件清单

### 需要读取的文件
1. `data/raw/训练集数据.csv` - 训练数据
2. `data/raw/验证集数据.csv` - 验证数据
3. `data/raw/测试集数据.csv` - 测试数据
4. `docs/Response模型训练方案_RFM完善版.md` - 建模方案
5. `docs/直播RFM优化后完整打分规则.md` - 直播 RFM 规则
6. `docs/字段含义.txt` - 字段说明
7. `/Users/yukewei/Public/xhs/RFM/rfm/rfm.py` - 历史 RFM 实现参考
8. `/Users/yukewei/Public/xhs/RFM/script/run_rfm.py` - 全平台 RFM 参考
9. `/Users/yukewei/Public/xhs/RFM/script/run_rfm_live.py` - 直播 RFM 参考

### 需要创建/修改的文件
1. `src/main.py` - 主入口（当前为空）
2. `src/config.py` - 配置管理
3. `src/data/loader.py` - 数据加载
4. `src/data/validator.py` - 数据校验
5. `src/features/rfm_platform.py` - 全平台 RFM
6. `src/features/rfm_live.py` - 直播 RFM
7. `src/features/feature_config.py` - 特征配置
8. `src/models/baseline.py` - 基线模型
9. `src/models/lgbm_trainer.py` - LightGBM 训练
10. `src/models/evaluator.py` - 评估器
11. `configs/config.yaml` - 配置文件
12. `requirements.txt` - 依赖包

## 依赖与顺序

### 实现顺序
1. **阶段 1**：项目结构 + 配置管理（无依赖）
2. **阶段 2**：数据加载与校验（依赖阶段 1）
3. **阶段 3**：双轨 RFM 打分（依赖阶段 2）
4. **阶段 4**：特征配置与工程（依赖阶段 3）
5. **阶段 5**：模型训练与评估（依赖阶段 4）
6. **阶段 6**：实验执行与结果落盘（依赖阶段 5）

### 模块依赖关系
```
main.py
  ├── config.py
  ├── data/loader.py
  │   └── data/validator.py
  ├── features/
  │   ├── rfm_platform.py
  │   ├── rfm_live.py
  │   └── feature_config.py
  └── models/
      ├── baseline.py
      ├── lgbm_trainer.py
      └── evaluator.py
```

## 潜在风险与缓解策略

### 风险 1：数据质量问题
- **风险描述**：143 个特征可能存在高空值率、异常值
- **缓解策略**：
  - 数据校验阶段全面检查空值、异常值
  - 对高空值特征（>50%）考虑剔除
  - 对数值特征做分位数检查，识别异常值

### 风险 2：标签漂移
- **风险描述**：正样本率从 Train 6.07% → Val 7.18% → Test 9.06%，存在跨集合漂移
- **缓解策略**：
  - 评估时同时报告全量用户和跨集合未重复用户两版指标
  - 阈值设定时考虑漂移因素
  - 使用 PR-AUC 而非 ROC-AUC 作为主指标（对不平衡数据更稳健）

### 风险 3：RFM 分位点不稳定
- **风险描述**：F/M 的三等分位点在不同集合可能差异较大
- **缓解策略**：
  - 严格使用训练集计算分位点
  - 验证集和测试集不重算分位点
  - 记录分位点并做敏感性分析（30 天、360 天窗口）

### 风险 4：模型过拟合
- **风险描述**：训练日期仅 5 天，时间覆盖窄，可能过拟合
- **缓解策略**：
  - 使用验证集早停（early stopping）
  - 使用特征子采样（feature_fraction, bagging_fraction）
  - 监控训练集和验证集指标差异
  - 考虑交叉验证（如果计算资源允许）

### 风险 5：特征穿越
- **风险描述**：某些特征可能包含未来信息
- **缓解策略**：
  - 严格检查所有特征的时间窗口
  - 确认所有特征严格截至 dtm 前
  - 特别关注含 `_yesterday` 的字段（元数据注明是 ds-16 天的状态）

## 代码片段参考

### 数据加载示例
```python
# src/data/loader.py
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """统一使用 utf-8-sig 处理 BOM"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df

def split_features_labels(df: pd.DataFrame, target_col: str, 
                          id_cols: list, label_blacklist: list):
    """分离主键、特征、标签"""
    # 剔除黑名单 Label
    feature_cols = [c for c in df.columns 
                    if c not in id_cols 
                    and c not in label_blacklist 
                    and c != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    ids = df[id_cols]
    
    return X, y, ids
```

### 直播 RFM 打分示例
```python
# src/features/rfm_live.py
def lifecycle_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """生命周期预分层：A/D/N/U"""
    df = df.copy()
    
    # A: F > 0
    df['lifecycle_segment'] = 'A'
    
    # D: F = 0 且 直播180天流失
    mask_d = (df['l180d_live_order_cnt'] == 0) & \
             (df['customer_stage_live_yesterday'] == '直播180天流失')
    df.loc[mask_d, 'lifecycle_segment'] = 'D'
    
    # N: F = 0 且 直播潜客
    mask_n = (df['l180d_live_order_cnt'] == 0) & \
             (df['customer_stage_live_yesterday'] == '直播潜客')
    df.loc[mask_n, 'lifecycle_segment'] = 'N'
    
    # U: 其他情况
    mask_u = (df['l180d_live_order_cnt'].isna()) | \
             (df['customer_stage_live_yesterday'] == 'unknown')
    df.loc[mask_u, 'lifecycle_segment'] = 'U'
    
    return df

def score_r(days: float) -> int:
    """R 打分：固定阈值"""
    if pd.isna(days) or days > 180:
        return None  # 标记为 U
    if days <= 7:
        return 3
    elif days <= 30:
        return 2
    else:
        return 1

def calculate_rfm_probability(df: pd.DataFrame, alpha: float = 100) -> pd.DataFrame:
    """训练集平滑响应率"""
    global_rate = df['target'].mean()
    
    rfm_prob = df.groupby('rfm_cell').agg({
        'target': ['sum', 'count']
    }).reset_index()
    rfm_prob.columns = ['rfm_cell', 'positive', 'sample']
    
    rfm_prob['rfm_probability'] = (
        (rfm_prob['positive'] + alpha * global_rate) / 
        (rfm_prob['sample'] + alpha)
    )
    
    return rfm_prob
```

### LightGBM 训练示例
```python
# src/models/lgbm_trainer.py
import lightgbm as lgb

def train_lgbm(X_train, y_train, X_val, y_val, params: dict):
    """训练 LightGBM"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model
```

### 评估指标示例
```python
# src/models/evaluator.py
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import numpy as np

def calculate_lift(y_true, y_pred, k_percent):
    """计算 Lift@K%"""
    n = len(y_true)
    k = int(n * k_percent / 100)
    
    # 按预测分数降序排序
    sorted_idx = np.argsort(-y_pred)[:k]
    y_true_sorted = y_true[sorted_idx]
    
    # 计算 Lift
    baseline_rate = y_true.mean()
    top_k_rate = y_true_sorted.mean()
    
    return top_k_rate / baseline_rate

def evaluate_model(y_true, y_pred):
    """统一评估"""
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred),
        'pr_auc': average_precision_score(y_true, y_pred),
        'logloss': log_loss(y_true, y_pred),
        'lift_1pct': calculate_lift(y_true, y_pred, 1),
        'lift_5pct': calculate_lift(y_true, y_pred, 5),
        'lift_10pct': calculate_lift(y_true, y_pred, 10),
        'lift_20pct': calculate_lift(y_true, y_pred, 20),
    }
    return metrics
```

## 验证步骤

### 阶段验证
1. **数据校验阶段**：
   - 检查样本量、正样本率是否与 README 一致
   - 检查跨集合用户重合度
   - 检查空值特征列表

2. **RFM 打分阶段**：
   - 检查全平台 RFM 分群分布是否合理
   - 检查直播 RFM 生命周期分层比例
   - 检查 RFM 响应率单调性（分数越高，响应率越高）

3. **模型训练阶段**：
   - 检查训练集和验证集指标差异（< 10% 为正常）
   - 检查特征重要性 Top 20 是否合理
   - 检查模型是否过拟合

4. **评估阶段**：
   - 检查所有实验的指标是否完整
   - 检查十分位分析是否单调
   - 检查 Lift@K 是否 > 1

### 最终验收
- 所有实验结果落盘到 `outputs/`
- 评估报告包含所有必需指标
- 代码可重复运行（固定随机种子）
- 文档完整（README、配置说明、运行指南）

## 总结

本实现计划覆盖了从数据加载到模型评估的完整流程，核心亮点包括：
1. **双轨 RFM 设计**：全平台 RFM 沿用历史项目思想但修正不稳健之处，直播 RFM 严格按文档实现
2. **统一评估框架**：所有实验在同一验证集/测试集上比较，确保公平性
3. **风险控制**：针对数据质量、标签漂移、模型过拟合等风险提出缓解策略
4. **可重复性**：固定随机种子、记录所有配置和阈值，确保结果可复现

预计实现时间：3-5 天（取决于调试和优化的深度）
