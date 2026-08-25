# 直播 RFM 优化后完整打分规则

## 1. 用途

该 RFM 用作 Response 模型的业务基线，预测目标与主模型保持一致：

```text
y = flag_repurchase_live_paid_14d
```

RFM 的所有输入必须为 `dtm` 当天打分时已经能够获得的历史数据。

## 2. 原始字段

```text
R = last_goods_live_buy_to_today_days
F = l180d_live_order_cnt
M = l180d_live_dgmv
生命周期参考 = customer_stage_live_yesterday
```

辅助核对字段：

```text
live_buy_active_flag
```

## 3. 第一步：生命周期预分层

不能简单将 `F=0` 全部定义为“无购买历史”。按以下优先级处理。

### A：近 180 天活跃购买用户

```text
F > 0
```

无论阶段字段为何值，只要近 180 天订单数大于 0，就进入 RFM 评分。

### D：历史购买但近 180 天沉睡用户

满足：

```text
F = 0
且 customer_stage_live_yesterday = "直播180天流失"
```

该人群单列为 `DORMANT`，不与真正无历史用户合并。

### N：无直播购买历史/直播潜客

满足：

```text
F = 0
且 customer_stage_live_yesterday = "直播潜客"
```

单列为 `NO_HISTORY`。

### U：口径未知或字段冲突用户

包括：

- `customer_stage_live_yesterday = "unknown"`；
- F、M、R 缺失；
- F=0，但阶段既不是直播潜客也不是直播180天流失；
- F>0，但 R 超出正常 180 天范围；
- 其他无法解释的字段冲突。

单列为 `UNKNOWN`，同时进入数据质量监控。

> 实际训练数据中“从未直播购买”的大值会随样本日期变化（训练集可见 2580～2584），因此实现中不写死 2603 等常数。生命周期以 `F` 和阶段字段联合判断；哨兵值只用于数据质量观察。
>
> 实际枚举还包含“直播老客>=3”和“直播新客1-2”。它们在 `F>0` 时统一按 A 类正常评分；仅当 `F=0` 时视为阶段与近 180 天订单冲突，归入 U 类。这样既覆盖主要用户，也不对冲突记录做无依据推断。

## 4. 第二步：活跃用户 RFM 打分

仅对生命周期为 A、即 `F>0` 的用户计算 R/F/M。

### R：最近直播购买距今天数

```text
0～7 天    → R=3
8～30 天   → R=2
31～180 天 → R=1
```

如果活跃用户出现 R>180 或缺失，标记为 `UNKNOWN`，不直接强行评分。

### F：近 180 天直播订单频次

在训练集的活跃用户中，对 `F>0` 计算 33.33% 和 66.67% 分位数：

```text
f_cut1 = Train 中 F 的 1/3 分位数
f_cut2 = Train 中 F 的 2/3 分位数
```

评分：

```text
1 ≤ F ≤ f_cut1       → F_score=1
f_cut1 < F ≤ f_cut2  → F_score=2
F > f_cut2           → F_score=3
```

当前 8000 条测试样本只用于分布诊断，其分位点约为 3 和 16，但不能把测试集分位点作为正式阈值。正式阈值必须由 7 月 10～14 日训练集确定。

处理整数重复边界：

- 不使用 `rank(method="first")`；
- 直接在训练集原值上计算分位点，使用左闭边界归档，保证相同原值始终同分；
- 去除重复分位点，若两个分位点相同则自动减少有效档数，不能按行顺序强行拆分同值用户；
- 阈值必须随模型及 `rule_version` 一并保存，验证集、测试集和线上预测均只加载，不重新计算。

### M：近 180 天直播成交金额

在训练集活跃用户中，对 `M>0` 计算三等分切点：

```text
m_cut1 = Train 中 M>0 的 1/3 分位数
m_cut2 = Train 中 M>0 的 2/3 分位数
```

评分：

```text
M ≤ m_cut1           → M_score=1
m_cut1 < M ≤ m_cut2  → M_score=2
M > m_cut2           → M_score=3
```

若 F>0 但 M=0，M_score 记为 1，并单独增加 `zero_gmv_with_order=1` 供质量检查。

当前测试样本的诊断分位点约为 278.43 和 1905.77，仅用于确认分布，不得用于正式评分。

## 5. 业务规则总分

对活跃用户：

```text
RFM_sum = R_score + F_score + M_score
```

范围：

```text
3～9
```

同时生成精细分群编码：

```text
RFM_cell = "A_R{R_score}F{F_score}M{M_score}"
```

示例：

```text
A_R3F2M1
```

非活跃用户不硬塞入 3～9 分体系：

```text
NO_HISTORY → lifecycle_segment=N
DORMANT    → lifecycle_segment=D
UNKNOWN    → lifecycle_segment=U
```

为兼容展示，可额外设置业务展示分：N=0、D=1、U=-1；但该展示分不能直接作为最终预测概率，也不能未经验证就认定 N 一定低于 D。

## 6. 用于模型对比的统一 RFM 预测分

由于 RFM_sum 等权求和、且非活跃用户无法自然排序，最终比较建议使用“训练集平滑响应率”。

### 分组键

```text
活跃用户：A_R{R}F{F}M{M}
沉睡用户：D
无历史用户：N
未知用户：U
```

### 训练集估计

对每个分组 g：

```text
rfm_probability(g)
= (positive_g + alpha × global_positive_rate)
  / (sample_g + alpha)
```

其中：

```text
positive_g = 训练集中该组正样本数
sample_g = 训练集中该组样本数
global_positive_rate = 训练集整体正样本率
alpha = 平滑强度
```

首版候选集合为 `{20, 50, 100, 200}`，以验证集 PR-AUC 选择 alpha；测试集不参与选择。全平台 RFM 分组也采用相同的训练集平滑响应率与验证集选 alpha 机制，使双轨 RFM 都能输出可比较的概率。

如果验证集或测试集出现训练集未见过的分组，预测分回退到：

1. 同生命周期大类的训练集正样本率；
2. 若仍不可用，再回退到训练集整体正样本率。

## 7. 阈值拟合与使用顺序

### 训练集：20260710～20260714

用于：

- 计算 `f_cut1`、`f_cut2`；
- 计算 `m_cut1`、`m_cut2`；
- 统计各 RFM 分组的平滑响应率；
- 保存完整规则配置。

### 验证集：20260726

使用训练集已冻结的阈值和响应率映射，用于：

- 选择平滑强度 alpha；
- 比较固定 RFM_sum、RFM_probability 和 Response 模型；
- 不重新计算 F/M 分位点。

### 测试集：20260802

只加载最终冻结配置并进行一次评估：

- 不重算阈值；
- 不重算分组响应率；
- 不再修改规则。

## 8. 输出字段

建议最终 RFM 结果保留：

```text
user_id
dtm
lifecycle_segment
r_raw
f_raw
m_raw
r_score
f_score
m_score
rfm_sum
rfm_cell
rfm_probability
rule_version
```

其中 `rule_version` 建议记录为：

```text
live_rfm_response_v1
```

## 9. 与 Response 主模型的比较

在同一验证集和测试集上比较：

1. 全局正样本率基线；
2. `RFM_sum`；
3. `RFM_probability`；
4. 143 个直播特征 LightGBM；
5. 143 个直播特征 + 4 个全平台 RFM 字段 LightGBM。

统一指标：

- ROC-AUC；
- PR-AUC；
- Lift@1%、5%、10%、20%；
- Top-K Precision / Recall；
- LogLoss、Brier Score；
- 十分位实际正样本率。

业务对比优先查看 PR-AUC、Lift@K 和 Top-K 实际直播支付复购率。
