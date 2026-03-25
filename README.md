# Kuramoto-PIGNN：面向复杂网络同步动力学与同步鲁棒性预测的物理引导图神经网络

本项目围绕 **Kuramoto 同步动力学（Kuramoto synchronization dynamics）**，构建了一套用于**复杂网络动力学建模**与**同步鲁棒性（synchronization robustness）预测**的物理引导图神经网络（Physics-Guided Graph Neural Network, PGNN）原型系统。

当前项目的核心目标不是传统的连通性鲁棒性预测，而是：

- 学习复杂网络上的**真实动力学演化**
- 在攻击场景下预测网络的**同步鲁棒性曲线**
- 比较纯数据驱动方法与物理引导方法在**长时 rollout**、**攻击鲁棒性预测**和**物理一致性**上的差异

---

## 1. 研究背景与问题定位

传统复杂网络鲁棒性预测方法很多是基于纯数据驱动的图神经网络（Graph Neural Network, GNN）直接做“图 -> 鲁棒性指标/曲线”的监督学习。但这类方法通常存在两个问题：

1. 缺少对底层动力学机制的显式约束  
2. 在长时闭环预测与攻击场景下容易出现漂移、失真或非物理行为

因此，本项目引入 **Kuramoto 动力学（Kuramoto dynamics）** 作为真实网络动力学主线，在图学习框架中显式注入动力学结构，构建更符合物理规律的预测模型。

本项目当前的研究主线为：

- **主线 A：真实网络动力学 + 物理引导图学习**  
  以 Kuramoto 同步动力学为基础，学习图上的真实动力学演化与同步鲁棒性

- **保留主线 B：连通性鲁棒性预测**  
  作为历史工作保留，但不再视为“真实动力学 PINN 主线”

- **后续副线 C：能控性鲁棒性预测**  
  未来扩展到有向生成式网络上的 controllability robustness

---

## 2. 当前任务定义

### 2.1 动力学建模任务

给定：

- 图 \( G=(V,E) \)
- 邻接矩阵 \( A \)
- 节点自然频率 \( \omega_i \)
- 初始相位 \( \theta_i(0) \)
- 耦合强度 \( K \)

学习 Kuramoto 动力学的一步演化与长时 rollout：

\[
\dot{\theta}_i(t)=\omega_i + K \sum_{j=1}^{N} A_{ij}\sin(\theta_j(t)-\theta_i(t))
\]

### 2.2 攻击场景下的同步鲁棒性预测

在节点攻击后，动力学变为：

\[
\dot{\theta}_i(t)=m_i(q)\left[\omega_i+K\sum_j A_{ij}m_j(q)\sin(\theta_j(t)-\theta_i(t))\right]
\]

其中 \(m_i(q)\) 为攻击掩码。

同步鲁棒性定义为 attacked rollout 最后若干步同步序参量 \(R(t)\) 的平均值：

\[
\mathcal{R}_{sync}(q)=\frac{1}{10}\sum_{t=T-9}^{T}R_q(t)
\]

---

## 3. 当前主要模型

本项目当前包含四类模型：

### 3.1 pure_data
纯数据驱动 GNN baseline，不加入显式动力学物理约束。

### 3.2 R_guided
在 pure_data 基础上加入图级同步量 \(R(t)\) 的监督/引导，希望增强对宏观同步行为的刻画。

### 3.3 v1
节点级动力学残差约束模型。  
特点是：在输出端加入 Kuramoto 动力学残差约束，但物理结构仍主要体现在节点级。

### 3.4 v2_edge（当前最优）
边消息物理对齐模型（edge-message physics-aligned model）。  
核心思想是把 Kuramoto 耦合项中的边级非线性作用：

\[
\sin(\theta_j-\theta_i)
\]

显式下放到边消息传递中，使模型在 message passing 阶段就能对真实耦合结构进行建模。

---

## 4. 项目目录结构

```text
kuramoto_pignn/
├── configs/                # 配置文件
├── data_generation/        # 图生成、Kuramoto 数值模拟、数据集构建
├── models/                 # 模型定义
├── physics/                # 动力学方程、物理残差相关实现
├── training/               # 损失函数与训练流程
├── scripts/                # 运行脚本、评估脚本、画图脚本
├── outputs/                # checkpoint、日志、图表输出（默认不纳入 Git）
├── README.md               # 项目说明
└── requirements.txt        # 依赖列表

5. 环境配置

建议使用独立 Python 环境运行本项目。

安装依赖：

pip install -r requirements.txt

如果使用 GPU，请确保：

已正确安装 CUDA
PyTorch 已支持 CUDA
torch.cuda.is_available() == True

当前实验主要在 GPU 环境下完成。

6. 数据集构建

运行以下命令构建 Kuramoto 数据集：

python scripts/run_build_kuramoto_dataset.py

数据构建流程包括：

生成 ER / BA / WS 图
采样节点自然频率 
𝜔
𝑖
∼
𝑁
(
0
,
1
)
ω
i
	​

∼N(0,1)
采样初始相位 
𝜃
𝑖
(
0
)
∼
𝑈
(
−
𝜋
,
𝜋
)
θ
i
	​

(0)∼U(−π,π)
使用数值积分生成 ground truth 动力学轨迹
构造图神经网络训练样本
7. 模型训练

训练主脚本：

python scripts/run_train_kuramoto_pignn.py

当前支持的模型标签包括：

pure_data
R_guided
v1
v2_edge
8. 动力学 rollout 评估
8.1 单图闭环评估
python scripts/rollout_eval.py

功能包括：

对单张图做闭环 rollout
绘制同步量 R(t) 曲线
绘制误差增长曲线
绘制代表节点相位轨迹
输出 summary 文件
8.2 批量 benchmark
python scripts/rollout_benchmark.py
python scripts/rollout_benchmark_all.py

功能包括：

对整个测试集做批量 rollout
输出 overall / topology / horizon 对照结果
比较不同模型在 ER / BA / WS 上的表现
9. 同步鲁棒性评估
9.1 随机攻击（Random Attack）
python -u scripts/evaluate_robustness.py --tags "pure_data,R_guided,v1,v2_edge" --split test --attack_mode random --q_values "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" --repeats 5 --rollout_steps 50 --tail_window 10 --device cuda --out_prefix outputs/logs/robustness_compare_all_random_test
9.2 度攻击（Degree Attack）
python -u scripts/evaluate_robustness.py --tags "pure_data,R_guided,v1,v2_edge" --split test --attack_mode degree --q_values "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" --repeats 1 --rollout_steps 50 --tail_window 10 --device cuda --out_prefix outputs/logs/robustness_compare_all_degree_test
10. 论文风格图表生成

当前项目支持自动生成 publication-style 图表：

python -u scripts/plot_publication_robustness.py --summary_paths outputs/logs/robustness_compare_all_random_test_summary.csv outputs/logs/robustness_compare_all_degree_test_summary.csv --detailed_paths outputs/logs/robustness_compare_all_random_test_detailed.csv outputs/logs/robustness_compare_all_degree_test_detailed.csv --attack_names random degree --topology_attack degree --save_dir outputs/figures/paper_ready

生成内容包括：

不同攻击协议下的 overall robustness curve
topology-wise robustness curve
robustness error / physics residual 对照柱状图
论文摘要表 paper_summary_table.csv
11. 当前核心实验结果
11.1 长时 rollout benchmark（test split, rollout 50 steps）
Model	PhaseMean	PhaseLast	RMean	RLast
pure_data	1.009400	1.302290	0.389930	0.518074
R_guided	0.981441	1.198089	0.319635	0.440253
v1	0.999297	1.271177	0.374637	0.505389
v2_edge	0.108973	0.130375	0.034647	0.021235

结论：

pure_data、R_guided 和 v1 在 one-step 上可以学习局部映射，但在长时闭环 rollout 中误差迅速累积
R_guided 对宏观同步量 R(t) 有一定帮助，但无法恢复真实微观动力学
v1 的节点级残差约束并未带来本质改善
v2_edge 在长时 rollout 上实现了明显、压倒性的提升
11.2 随机攻击下的同步鲁棒性预测（Random Attack）

在 random attack 下，v2_edge 能稳定重建 ODE ground truth 的同步鲁棒性曲线。

以 overall 结果为例：

q	TrueRob	PredRob (v2_edge)	AbsErr
0.00	0.792575	0.770808	0.022149
0.10	0.756593	0.739572	0.022092
0.30	0.659206	0.633503	0.029732
0.50	0.536033	0.515363	0.029932

结果特征：

v2_edge 在 random attack 下整体贴近 ground truth
误差大体稳定在 0.02–0.03 量级
相比之下，pure_data、R_guided 和 v1 的预测曲线普遍偏低，且形状明显失真
说明 v2_edge 不只是未攻击图上 rollout 更准确，也能在随机攻击场景下保持对同步鲁棒性的有效预测