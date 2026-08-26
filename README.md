# 券敏感人群第一阶段 — Response 模型训练

## 项目简介

本项目为券敏感人群项目的**第一阶段：Response 模型训练**。

- **建模目标**：以 `user_id × dtm` 为样本粒度（样本域为当日下单用户），预测用户在未来 14 天内是否发生**使用平台券的直播复购**
- **目标标签**：`flag_repurchase_live_paid_14d`（`paid` 表示复购时使用了平台券，即券敏感口径；对照口径 `flag_repurchase_live_14d` 为不限是否用券的直播复购）
- **主模型特征**：143 个样本日前可获得的直播历史行为特征
- **业务基线（双轨 RFM）**：
  - 全平台 RFM：沿用历史项目方案（`/Users/yukewei/Public/xhs/RFM`：五分位打分 + 10 类分层，R=`l720d_last_to_today_days` / F=`l360d_buy_days` / M=`l360d_dgmv`，仅 `total_orders >= 3` 成熟客户参与打分）
  - 直播 RFM：按 `docs/直播RFM优化后完整打分规则.md`（三档打分 + A/D/N/U 生命周期预分层 + 训练集平滑响应率）
- **建模方案**：详见 `docs/Response模型训练方案_RFM完善版.md`

**当前状态**：双轨 RFM、数据校验、统一评估及 Response 训练代码已实现，可执行验证集选模；测试集仅在显式指定参数时读取和评估。

## 目录结构

```text
券敏感人群第一阶段/
├── data/
│   ├── raw/            # 原始数据：上游数仓直接导出的训练/验证/测试集
│   └── processed/      # 处理后数据（预留）：特征工程、清洗后的产出
├── configs/            # 训练配置与超参数
├── src/                # 双轨 RFM、训练与评估代码
├── checkpoints/        # 模型权重与训练检查点（预留）
├── logs/               # 训练日志（预留）
├── outputs/            # 评估结果与生成样本（预留）
├── docs/               # 项目文档与实验记录
├── tests/              # 测试代码（预留）
├── notebooks/          # Jupyter 笔记本（预留）
├── to_sort/            # 待确认归类的文件
└── README.md           # 本文件
```

## 各目录说明

| 目录 | 用途 | 当前内容 |
|---|---|---|
| `data/raw/` | 上游直接导出的原始数据，只读、不做任何修改 | 训练集/验证集/测试集 CSV |
| `data/processed/` | 经清洗、特征工程处理后的数据 | 空（预留） |
| `configs/` | 训练配置、超参数文件 | `training_config.json` |
| `src/` | 数据校验、双轨 RFM、Response 训练与统一评估 | `main.py`、`rfm.py`、`evaluation.py` |
| `checkpoints/` | 模型权重、训练检查点 | 空（预留） |
| `logs/` | 训练与评估日志 | 空（预留） |
| `outputs/` | 评估指标、预测结果、生成样本 | 空（预留） |
| `docs/` | 建模方案、打分规则、字段说明等文档 | 2 份方案文档 + 完整字段说明 |
| `tests/` | 单元测试与集成测试 | 空（预留） |
| `notebooks/` | 探索性分析（EDA）笔记本 | 空（预留） |
| `to_sort/` | 暂未明确归类的文件，确认后移出 | `方案选择`（空文件） |

## 数据说明（data/raw/）

| 文件 | 大小 | 行数（含表头） | 日期范围 |
|---|---|---|---|
| 训练集数据.csv | 899MB | 867,476 | 20260710～20260714 |
| 验证集数据.csv | 213MB | 167,556 | 20260726 |
| 测试集数据.csv | 168MB | 161,719 | 20260802 |

**列结构（159 列）**：3 主键/标识 + 9 Label + 4 个全平台 RFM 字段（`l720d_last_to_today_days`、`l360d_buy_days`、`l360d_dgmv`、`total_orders`，位于第 12~15 列）+ 143 直播特征。注意：字段含义文档仅列出 155 列（未含全平台 4 字段），实际 CSV 已包含——方案中 E5 实验（143 直播特征 + 4 全平台字段）**无需额外补取数据**。

**正样本率实测**：

| 口径 | Train | Validation | Test |
|---|---|---|---|
| `flag_repurchase_live_paid_14d`（本方案 y） | 6.07% | 7.18% | 9.06% |
| `flag_repurchase_live_14d`（对照口径） | 29.89% | 31.57% | 31.00% |

注意：`live_paid` 口径正样本率逐期上升（6.07% → 7.18% → 9.06%），存在跨集合漂移，评估与阈值设定时需考虑。

**读取注意事项**：三个 CSV 的导出格式不一致——

- `训练集数据.csv`：无 BOM，字段带双引号
- `验证集数据.csv`：带 UTF-8 BOM，字段不带双引号
- `测试集数据.csv`：带 UTF-8 BOM，字段带双引号

建议读取时统一使用 `encoding='utf-8-sig'` 以兼容 BOM 差异。

**建模提醒**（来自 `docs/Response模型训练方案_RFM完善版.md`）：

- 训练日期仅覆盖 5 天，时间覆盖较窄
- 14 天标签观察窗在相邻集合间存在重叠，验证/测试并非完全独立
- 需统计各集合样本量、正样本率及跨集合重复 `user_id` 比例
- Validation 用于选模调参，Test 最终只使用一次

## RFM 规则与训练代码

### 全平台 RFM

- R/F/M：`l720d_last_to_today_days` / `l360d_buy_days` / `l360d_dgmv`
- 生命周期：0 单单列 N，1～2 单单列 NEW，`total_orders >= 3` 进入成熟用户评分
- 成熟用户按训练集原值五分位打 1～5 分；R 越小分越高，F/M 越大分越高
- 不使用 `rank(method="first")`，相同原值用户始终同分；重复分位点自动减少有效档数
- 保留历史 10 类业务分群，并为所有生命周期/评分单元计算训练集平滑响应率

### 直播 RFM

- R/F/M：`last_goods_live_buy_to_today_days` / `l180d_live_order_cnt` / `l180d_live_dgmv`
- 生命周期：A（F>0 且 R 合法）、D（180 天流失且 F=0）、N（直播潜客且 F=0）、U（缺失或冲突）
- R 固定阈值：0～7/8～30/31～180 天对应 3/2/1 分；F/M 用训练集三等分位
- 未购买哨兵值随日期变化，不写死具体常数；F=0 的老客或新客冲突记录归 U 并监控
- 平滑强度 alpha 从 `{20, 50, 100, 200}` 中按验证集 PR-AUC 选择

### 实验

- E0：训练集全局正样本率
- E1：直播 RFM 业务排序分
- E2：直播与全平台 RFM 平滑响应率
- E3：双轨 RFM 原始字段轻量 LightGBM
- E4：143 个直播特征 LightGBM
- E5：143 个直播特征 + 4 个全平台 RFM 字段 LightGBM

安装依赖并训练（默认只读取 Train/Validation，不触碰 Test）：

```bash
python3 -m pip install -r requirements.txt
python3 src/main.py
```

训练完成后会在 `checkpoints/<run-id>/frozen_manifest.json` 固化验证集选出的唯一最佳模型，并保存配套预处理器、RFM 规则和已见用户摘要。测试阶段必须明确指定该训练 run-id，程序只加载冻结产物，不重新训练、不重新选参：

```bash
# 可先校验所有冻结文件能否加载；不读取 Test
python src/main.py --evaluate-test --run-id 20260825_162323 --dry-run

# 确认后执行一次正式 Test
python src/main.py --evaluate-test --run-id 20260825_162323
```

测试结果写入 `outputs/<run-id>/test/`。若该目录已存在 `test_metrics.csv`，程序会拒绝重复评估。`--evaluate-test` 不指定 `--run-id` 也会直接报错，防止误选 checkpoint。

开发阶段可使用 `--sample-rows 20000` 做端到端冒烟验证，或使用 `--skip-model` 只检查数据和 RFM 基线。指标输出到 `outputs/<时间戳>/`，模型、预处理器与冻结规则输出到 `checkpoints/<时间戳>/`，日志输出到 `logs/`。

## 如何 Review 一次训练

不要逐个打开 `outputs/<时间戳>/` 下的 CSV。先为目标运行生成 Review 页面：

```bash
python src/visualize_results.py --run-id 20260825_162323
```

日常只需依次查看 `outputs/<时间戳>/review/` 下的文件：

1. `00_先看这里_结果总览.html`：结论、完整实验指标和最佳模型位置；
2. `01_实验效果对比.png`：比较 ROC-AUC 与 AP；
3. `02_TopK指标对比.png`：比较 K=1%/5%/10%/20% 下的 Precision、Recall 和 Lift；
4. `03_十分位响应率.png`：判断高分到低分人群的实际响应率是否形成稳定梯度；
5. `04_最佳模型特征重要性.png`：理解模型主要使用哪些特征，不作因果解释；
6. `05_RFM分布漂移与响应率.png`：检查直播和全平台生命周期的 Train/Validation 漂移及响应率。

只有排查问题时才看上级目录的底层文件：`validation_metrics.csv` 用于核对完整指标，`validation_*_deciles.csv` 用于检查分档细节，`*_feature_importance.csv` 用于查看全部特征，`rfm_rules_and_quality.json` 用于核对冻结规则。

当前冻结并推荐用于最终 Test 的完整运行是 `20260825_162323`；其他时间目录主要是开发冒烟、RFM 校验或中间调参记录，暂时无需查看。

## 如何 Review Test 结果

Test 完成后，基于已冻结的 run-id 生成 Test Review 报告（不重新运行 Test、不训练模型）：

```bash
python src/visualize_test_results.py --run-id 20260825_162323
```

报告生成在 `outputs/<run-id>/test/review/` 下：

1. `00_test_review.html`：Test 评估总览，含核心结论、指标明细、部署建议
2. `01_validation_vs_test.png`：Validation vs Test 指标对比，观察泛化能力
3. `02_all_vs_unseen_users.png`：全量用户 vs Unseen Users 对比，评估对新用户的预测能力
4. `03_test_deciles.png`：Test 十分位响应率，判断模型排序可靠性
5. `04_calibration.png`：概率校准分析（网页附录）
6. `05_positive_rate_drift.png`：正样本率漂移分析（网页附录）

**关键发现**（run-id `20260825_162323`）：
- Test 正样本率（9.06%）高于 Validation（7.18%），会抬高随机基线 AP，因此 AP 绝对值需结合 ROC-AUC 与 Lift 解读；
- E5 在 Test 全量用户上 ROC-AUC=0.9003、AP=0.5427、Lift@1%=9.30，Top 1% 实际命中率=84.30%；
- Unseen Users 正样本率为 3.66%，E5 ROC-AUC=0.8724、AP=0.2749、Lift@1%=13.02；低 AP 部分来自更低基础率，不能单独断言泛化失效；
- Test 证明了时间外相关性排序能力，但不能直接证明发券增量收益；上线前仍需阈值/预算评审及随机实验或 uplift/因果评估；
- 若要将模型分数解释为真实概率，而不仅用于排序，需要使用独立校准集或后续时间窗进行概率校准。

## 文件整理记录（2026-08-24）

| 原路径 | 新路径 |
|---|---|
| `训练集数据.csv` | `data/raw/训练集数据.csv` |
| `验证集数据.csv` | `data/raw/验证集数据.csv` |
| `测试集数据.csv` | `data/raw/测试集数据.csv` |
| `main.py` | `src/main.py` |
| `Response模型训练方案_RFM完善版.md` | `docs/Response模型训练方案_RFM完善版.md` |
| `直播RFM优化后完整打分规则.md` | `docs/直播RFM优化后完整打分规则.md` |
| `字段含义.txt` | `docs/字段含义.txt` |
| `方案选择` | `to_sort/方案选择`（空文件，待确认归类） |
