# Kuramoto-PIGNN：复杂网络同步动力学与同步鲁棒性预测

本项目围绕 **Kuramoto 同步动力学**，构建了一套用于复杂网络动力学建模与同步鲁棒性预测的**物理引导图神经网络（Physics-Guided Graph Neural Network, PGNN）**原型系统。

当前项目的核心目标包括：

- 学习复杂网络上的真实同步动力学演化
- 比较纯数据驱动方法与物理引导方法在长时闭环 rollout 中的表现差异
- 在节点攻击场景下预测网络的同步鲁棒性曲线
- 分析预测精度、长时稳定性与物理残差之间的关系

---

## 1. 项目背景

传统复杂网络鲁棒性预测方法很多是基于纯数据驱动的图神经网络（GNN）直接做“图 -> 鲁棒性指标/曲线”的监督学习。但这类方法通常存在两个问题：

- 缺少对底层动力学机制的显式约束
- 在长时闭环 rollout 与攻击场景下容易出现漂移、失真或非物理行为

因此，本项目选择 **Kuramoto 动力学** 作为真实网络动力学主线，在图学习框架中显式引入动力学结构，希望同时提升：

- 长时预测精度
- 攻击场景下同步鲁棒性预测能力
- 预测轨迹对真实动力学方程的一致性

---

## 2. 当前任务定义

### 2.1 动力学建模任务
给定图 $G=(V,E)$、邻接矩阵 $A$、节点自然频率 $\omega_i$、初始相位 $\theta_i(0)$ 以及耦合强度 $K$，学习 Kuramoto 动力学的一步演化与长时 rollout：

$$\dot{\theta}_i(t)=\omega_i + K \sum_{j=1}^{N} A_{ij}\sin(\theta_j(t)-\theta_i(t))$$

### 2.2 攻击场景下的同步鲁棒性预测
在节点攻击后，引入攻击掩码 $m_i(q)$，动力学变为：

$$\dot{\theta}_i(t)=m_i(q)\left[\omega_i+K\sum_j A_{ij}m_j(q)\sin(\theta_j(t)-\theta_i(t))\right]$$

同步鲁棒性定义为 attacked rollout 最后若干步同步序参量 $R(t)$ 的平均值：

$$\mathcal{R}_{sync}(q)=\frac{1}{10}\sum_{t=T-9}^{T}R_q(t)$$

---

## 3. 当前主要模型

本项目当前包含四类模型：

- **pure_data**：纯数据驱动 GNN baseline
- **R_guided**：加入图级同步量 \( R(t) \) 引导的模型
- **v1**：节点级动力学残差约束模型
- **v2_edge**：边消息物理对齐模型（当前最优）

其中，`v2_edge` 的核心思想是把 Kuramoto 耦合项中的边级非线性作用显式下放到 message passing 阶段进行建模，从而更好地学习真实耦合动力学。

---

## 4. 项目目录结构

```text
kuramoto_pignn/
├── configs/                # 配置文件
├── data_generation/        # 图生成、Kuramoto 数值模拟、数据集构建
├── models/                 # 模型定义
├── physics/                # 动力学方程与物理残差相关实现
├── training/               # 损失函数与训练流程
├── scripts/                # 运行脚本、评估脚本、画图脚本
├── outputs/                # checkpoint、日志、图表输出（默认不纳入 Git）
├── README.md               # 项目说明
└── requirements.txt        # 依赖列表
```

---

## 5. 环境配置

建议使用独立 Python 环境运行本项目。

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 GPU，请确保：

- 已正确安装 CUDA
- PyTorch 已支持 CUDA
- `torch.cuda.is_available() == True`

当前实验主要在 GPU 环境下完成。

---

## 6. 数据集构建

运行以下命令构建 Kuramoto 数据集：

```bash
python scripts/run_build_kuramoto_dataset.py
```

数据构建流程包括：

- 生成 ER / BA / WS 图
- 采样节点自然频率 \( \omega_i \sim \mathcal{N}(0,1) \)
- 采样初始相位 \( \theta_i(0)\sim U(-\pi,\pi) \)
- 使用数值积分生成 ground truth 动力学轨迹
- 构造图神经网络训练样本

---

## 7. 模型训练

训练主脚本：

```bash
python scripts/run_train_kuramoto_pignn.py
```

当前支持的模型标签包括：

```text
pure_data
R_guided
v1
v2_edge
```

---

## 8. 动力学 rollout 评估

### 8.1 单图闭环评估

```bash
python scripts/rollout_eval.py
```

### 8.2 批量 benchmark

```bash
python scripts/rollout_benchmark.py
python scripts/rollout_benchmark_all.py
```

可用于：

- 对整个测试集做批量 rollout 评估
- 输出 overall / topology / horizon 对照结果
- 比较不同模型在 ER / BA / WS 上的表现

---

## 9. 同步鲁棒性评估

### 9.1 Random attack

```bash
python -u scripts/evaluate_robustness.py --tags "pure_data,R_guided,v1,v2_edge" --split test --attack_mode random --q_values "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" --repeats 5 --rollout_steps 50 --tail_window 10 --device cuda --out_prefix outputs/logs/robustness_compare_all_random_test
```

### 9.2 Degree attack

```bash
python -u scripts/evaluate_robustness.py --tags "pure_data,R_guided,v1,v2_edge" --split test --attack_mode degree --q_values "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" --repeats 1 --rollout_steps 50 --tail_window 10 --device cuda --out_prefix outputs/logs/robustness_compare_all_degree_test
```

---

## 10. 高质量图表生成

```bash
python -u scripts/plot_publication_robustness.py --summary_paths outputs/logs/robustness_compare_all_random_test_summary.csv outputs/logs/robustness_compare_all_degree_test_summary.csv --detailed_paths outputs/logs/robustness_compare_all_random_test_detailed.csv outputs/logs/robustness_compare_all_degree_test_detailed.csv --attack_names random degree --topology_attack degree --save_dir outputs/figures/paper_ready
```

生成内容包括：

- 不同攻击协议下的 overall robustness curve
- topology-wise robustness curve
- robustness error / physics residual 对照柱状图
- 汇总表 `paper_summary_table.csv`

---

## 11. 当前核心实验结果

### 11.1 长时 rollout benchmark（test split, rollout 50 steps）

| Model | PhaseMean | PhaseLast | RMean | RLast |
|---|---:|---:|---:|---:|
| pure_data | 1.009400 | 1.302290 | 0.389930 | 0.518074 |
| R_guided | 0.981441 | 1.198089 | 0.319635 | 0.440253 |
| v1 | 0.999297 | 1.271177 | 0.374637 | 0.505389 |
| v2_edge | **0.108973** | **0.130375** | **0.034647** | **0.021235** |

**结果分析：**

- `pure_data`、`R_guided` 和 `v1` 在 one-step 上可以学习局部映射，但在长时闭环 rollout 中误差迅速累积
- `R_guided` 对宏观同步量 \( R(t) \) 有一定帮助，但无法恢复真实微观动力学
- `v1` 的节点级残差约束没有带来本质改善
- `v2_edge` 在长时 rollout 上实现了明显、压倒性的提升

---

### 11.2 随机攻击下的同步鲁棒性预测（Random Attack）

以 overall 结果为例：

| q | TrueRob | PredRob (v2_edge) | AbsErr |
|---|---:|---:|---:|
| 0.00 | 0.792575 | 0.770808 | 0.022149 |
| 0.10 | 0.756593 | 0.739572 | 0.022092 |
| 0.30 | 0.659206 | 0.633503 | 0.029732 |
| 0.50 | 0.536033 | 0.515363 | 0.029932 |

**结果分析：**

- `v2_edge` 在 random attack 下整体贴近 ground truth
- 误差大体稳定在 **0.02–0.03** 量级
- 相比之下，`pure_data`、`R_guided` 和 `v1` 的预测曲线普遍偏低，且形状明显失真
- 说明 `v2_edge` 不只是未攻击图上 rollout 更准确，也能在随机攻击场景下保持对同步鲁棒性的有效预测

---

### 11.3 度攻击下的同步鲁棒性预测（Degree Attack）

在 degree attack 下，各模型按 q 平均后的 overall robustness 绝对误差约为：

| Model | Mean Robustness AbsErr |
|---|---:|
| pure_data | 0.275942 |
| R_guided | 0.223318 |
| v1 | 0.265347 |
| v2_edge | **0.022568** |

**结果分析：**

- `v2_edge` 在 degree attack 下的 robustness 误差比其他模型低约一个数量级
- 这说明其优势不仅体现在随机攻击场景，也体现在更强的 targeted attack 中
- 说明将物理结构显式注入边消息传递，有助于增强模型对攻击后动力学退化过程的建模能力

---

### 11.4 Prediction physics residual

本项目在同步鲁棒性评估中额外监控了推理阶段物理残差。

**观察结果：**

- `pure_data`、`R_guided` 和 `v1` 的 prediction physics residual 普遍在 **1.0–2.5** 量级
- `v2_edge` 的 prediction physics residual 在 random / degree 两种攻击下稳定在 **0.05–0.10** 左右

**说明：**

`v2_edge` 的优势并不是简单的经验拟合，而是同时体现为：

- 更准确的同步鲁棒性曲线预测
- 更低的长时 rollout 误差
- 更强的动力学物理一致性

---

## 12. 当前阶段性结论

基于当前结果，可以得到以下结论：

1. 基于 Kuramoto 动力学构建真实网络动力学主线是可行的
2. pure-data GNN 难以稳定学习攻击下的同步动力学退化过程
3. graph-level 引导与节点级残差约束只能提供有限帮助
4. 将物理结构显式注入边消息传递的 `v2_edge`，能够同时提升：
   - 长时 rollout 精度
   - 同步鲁棒性曲线预测精度
   - 动力学物理一致性
5. 当前结果说明，`v2_edge` 是本项目现阶段最有效的模型设计

---

## 13. 已完成的阶段性工作

当前已完成：

- Kuramoto 数据集构建
- `pure_data / R_guided / v1 / v2_edge` 四模型训练
- 单图与批量 rollout benchmark
- random attack 下同步鲁棒性预测
- degree attack 下同步鲁棒性预测
- prediction physics residual 监控
- publication-style 图表生成
- Git 版本管理与 GitHub 远程仓库建立

当前阶段性标签：

```text
v_sync_robustness_stage1
```

---

## 14. 后续计划

后续计划包括：

- 加入 betweenness attack
- 进一步诊断轻微保守偏差来源
- 分析高 q 场景下不同拓扑的误差机制
- 完善实验图表与论文写作
- 后续扩展到更广义的网络功能鲁棒性与能控性鲁棒性场景

---

## 15. 说明

本项目当前仍在持续研究与迭代阶段。  
代码结构、实验脚本与结果整理将随着后续研究推进继续更新。