# 阶段一实现 Review 与收尾建议

## 一、结论

当前工程已经具备完整的训练骨架：数据校验、双轨 RFM、E0～E5 对照实验、时间外验证、模型保存和 Review 页面都已实现。问题不在于“缺少产物”，而在于开发过程产物与阶段交付产物混在一起，导致 reviewer 需要从多个运行目录和二十多个平铺文件中自行拼结论。

用户提出“阶段最后应由一次实验完成指标评价并给出结论”的方向基本正确，但应区分：

- 一次**最终实验运行/实验套件**：正确，应有唯一 run_id、冻结配置和明确结论；
- 只保留一个输出文件：不正确，模型、预处理器、配置、指标、审计信息仍需保留以便复现；
- reviewer 只看一个入口：正确，应提供一个报告页，其余文件下沉为 artifacts。

## 二、当前结果是否已经可以结题

暂时不能直接结题，原因按优先级排列如下。

### 已确认：目标标签业务语义成立

用户已向数据负责人核实：`flag_repurchase_live_paid_14d` 中的 `paid` 表示使用平台券（平台补贴）。因此当前项目可准确表述为：

> 预测当日下单用户未来 14 天是否发生使用平台券的直播复购。

README、图表和报告中的平台券口径可以保留。建议在字段说明或项目 README 中记录该确认来源与日期，避免后续再次产生歧义。

### P0：尚无正式全量测试结果

- 完整运行 `20260825_101820` 只包含 Train + Validation；
- 包含 Test 的 `20260825_101742` 对 Train/Validation/Test 均只读取前 5000 行，是冒烟测试；
- 因此阶段一目前完成了训练和验证，但没有完成全量 Test 终评。

另外，Test 已被用于开发期抽样查看。若追求严格盲测，应使用后续新的完整分区作为最终 Test；若暂时只能使用 20260802，则应在报告中称为“开发期已查看的时间外 Holdout”，避免声称完全未触碰。

### P1：E5 相比 E4 的增益非常小

验证集：

- E4：PR-AUC 0.5051，ROC-AUC 0.9076；
- E5：PR-AUC 0.5065，ROC-AUC 0.9080；
- PR-AUC 绝对提升仅 0.0014，Lift@1% 完全相同（11.17x）。

因此不能仅凭点估计就下结论“E5 显著优于 E4”。建议做 paired bootstrap 或在完整时间外 Test 上比较。若差异仍极小，应从部署成本、字段稳定性和解释性考虑是否采用更简单的 E4。

### P1：类别特征当前被当作有序数字

`OrdinalEncoder` 将类别编码为整数，但 LightGBM 训练时没有声明 categorical features，模型会把编码值当连续数值切分，产生并不存在的类别大小顺序。

建议：

- 优先使用 LightGBM 原生 categorical feature；或
- 对低基数类别做 one-hot；高基数类别使用经过严格训练集拟合的编码方案。

修正后需要重新跑验证，当前结果可以作为阶段性结果，但不宜直接作为最终模型冻结。

### P1：最终测试入口应加载冻结模型，而不是重新训练

当前 `python3 src/main.py --evaluate-test` 会重新读取 Train/Validation、重新搜索候选参数、重新训练模型，再评估 Test。更稳妥的终评流程应是：

```text
train.py       → 生成冻结 run_id
finalize.py    → 固化选中模型与规则
evaluate.py    → 只加载该 run_id 的模型、预处理器和规则，读取 Test 一次
```

这样能证明 Test 评估使用的就是验证阶段冻结的模型。

## 三、当前结果能说明什么

在目标字段暂按“直播付费复购”理解的前提下：

1. RFM 基线有效：直播 RFM PR-AUC 0.2936，明显高于全局正样本率 0.0718；
2. 双轨 RFM 原始字段轻量模型达到 PR-AUC 0.4611；
3. 143 个直播特征模型达到 PR-AUC 0.5051，说明漏斗、交易和活跃特征提供额外信息；
4. E5 验证集 PR-AUC 0.5065，ROC-AUC 0.9080，Top 10% 实际正样本率 42.97%，相对总体约 5.99 倍；
5. E5 十分位实际正样本率从 42.97% 单调下降到 0.066%，排序结构较好；
6. 特征重要性高度集中于历史直播 DGMV：`l180d_live_dgmv`、`l90d_live_dgmv`、`l60d_live_dgmv` 等，需要继续确认时间口径无穿越；
7. Validation 中 36.53% 用户在 Train 出现过。虽然 `user_id` 未输入模型，但最终报告应同时给出全量与未见用户指标。

## 四、为什么现在显得杂乱

1. `outputs/` 下有 10 个开发运行目录，冒烟测试、RFM 校验、调参和完整训练混在一起；
2. 每个 alpha 都生成一份 decile CSV，调参细节占据主目录；
3. `outputs/`、`checkpoints/`、`logs/` 按同一 run_id 分散在三个位置；
4. 原始指标、可视化、summary.txt、review HTML 重复表达同一结论；
5. `main_optimized.py` 已从 `src/` 移至 `to_sort/` 作为历史优化副本；当前唯一正式训练入口为 `src/main.py`；
6. `to_sort/方案选择` 是空文件；
7. README 的目录现状描述已过期，并包含待确认的标签语义。

## 五、推荐的最终目录

```text
券敏感人群第一阶段/
├── README.md
├── configs/
│   └── phase1_final.json
├── data/
│   └── raw/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── rfm.py
│   ├── metrics.py
│   └── report.py
├── tests/
├── runs/
│   ├── archive/                    # 历史开发/冒烟运行
│   └── phase1_final_<run_id>/
│       ├── report.html             # reviewer 唯一入口
│       ├── conclusion.md           # 结论与限制
│       ├── metrics.csv             # 仅选中配置后的对比表
│       ├── model/
│       │   ├── model.txt
│       │   ├── preprocessor.pkl
│       │   └── metadata.json
│       └── artifacts/              # 默认无需打开
│           ├── tuning/
│           ├── deciles/
│           ├── feature_importance/
│           ├── data_audit/
│           ├── resolved_config.json
│           ├── run_manifest.json
│           └── train.log
└── docs/
```

重点不是减少底层文件总数，而是将 reviewer 的入口减少为：

1. `report.html`；
2. `conclusion.md`；
3. 必要时查看 `metrics.csv`。

## 六、最终实验应如何定义

建议把“最终实验”定义为一次预注册的 benchmark suite，而不是只跑一个模型：

- E0：全局正样本率；
- E2-Live：直播 RFM；
- E2-Platform：全平台 RFM；
- E3：双轨 RFM 轻量模型；
- Candidate A：E4；
- Candidate B：E5。

Validation 负责选择 E4/E5 和所有超参数。冻结后，最终 Test 只评估：

- E0；
- 两个 RFM 基线；
- 最终选中的一个 Response 模型。

最终报告只保留选中 alpha 后的结果，alpha 搜索明细和候选超参数进入 `artifacts/tuning/`。

## 七、推荐收尾顺序

1. 在字段说明或 README 中记录已确认的 `paid = 使用平台券（平台补贴）` 口径；
2. 修正类别特征处理方式；
3. 明确 E4/E5 选择规则，增加 paired bootstrap 或以完整 Test 结果判断；
4. 将 Test 入口改为加载冻结模型；
5. 运行一次新的完整 Validation 训练；
6. 使用未触碰的新分区做正式 Test；若只能用 20260802，则明确其为已查看 Holdout；
7. 自动生成单一 `report.html + conclusion.md + metrics.csv`；
8. 将其余历史运行移动到 `runs/archive/`，不删除，以便追溯。
