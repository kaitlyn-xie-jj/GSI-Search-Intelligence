# GSI Search Intelligence：论文参考文献与研究工作缺口

> 目的：为当前仓库发展成论文建立可辩护的 reference map。审计日期：2026-08-04。
>
> 仓库目前没有正式 bibliography。本文区分“方法基础”“高度相关工作”“工程项目”和“专利 prior art”，不宣称仓库复现了未实际采用的方法。

## 1. 当前工作的学术定位

当前 Search Intelligence 可概括为：由 LLM 产生 task-conditioned semantic spatial prior，以离散 Bayesian belief 表示目标位置不确定性，并由“检测概率 + 期望信息增益 + 未观测新颖性 − 运动代价”选择下一视点的闭环主动搜索框架。

相关方向包括 ObjectGoal Navigation、semantic exploration、Bayesian target search、active visual search、Next-Best-View（NBV）、Coverage Path Planning（CPP）、semantic mapping、open-vocabulary detection、POMDP 和 embodied navigation evaluation。

## 2. 核心必引文献

### R1. PONI: Potential Functions for ObjectGoal Navigation With Interaction-Free Learning

- 信息：Santhosh Kumar Ramakrishnan, Devendra Singh Chaplot, Ziad Al-Halah, Jitendra Malik, Kristen Grauman. CVPR 2022, 18890–18900. arXiv:2201.10029。
- 论文：https://openaccess.thecvf.com/content/CVPR2022/html/Ramakrishnan_PONI_Potential_Functions_for_ObjectGoal_Navigation_With_Interaction-Free_Learning_CVPR_2022_paper.html
- 开源：https://github.com/srama2512/PONI
- 内容：将 ObjectNav 中“去哪里找”与“如何到达”解耦；从 top-down semantic map 预测 object potential 和 area potential，用于选择 frontier long-term goal；通过被动地图数据做 interaction-free supervised learning。
- 相关性：本仓库同样用语义地图回答“下一步去哪里看”，并把视点决策与底层运动解耦。
- 区别：本仓库没有 learned potential-function network、frontier supervision 和 ObjectNav local policy；当前是 LLM 标签权重投影 + Bayesian belief + 手工多目标 utility。因此应列为最重要相关工作，但不能声称复现 PONI。

### R2. Object Goal Navigation using Goal-Oriented Semantic Exploration（SemExp）

- 信息：Devendra Singh Chaplot, Dhiraj Prakashchand Gandhi, Abhinav Gupta, Ruslan Salakhutdinov. NeurIPS 2020, Vol. 33. arXiv:2007.00643。
- 论文：https://papers.nips.cc/paper_files/paper/2020/hash/2c75cf2681788adaca63aa95ae028b22-Abstract.html
- 开源：https://github.com/devendrachaplot/Object-Goal-Navigation
- 内容：构建 episodic semantic map，用目标类别条件化长期策略学习物体空间关系先验，再由 local policy 导航。
- 相关性：对应仓库的 SemanticGridBuilder、task-conditioned prior 和“语义决策/底层导航”分层。
- 区别：SemExp 的先验由 RL policy 学得；本仓库由 LLM 给标签相关性，再显式维护 Bayesian target belief。

### R3. Probabilistic Robotics

- 信息：Sebastian Thrun, Wolfram Burgard, Dieter Fox. MIT Press, 2005. ISBN 9780262201629。
- 链接：https://mitpress.mit.edu/9780262201629/probabilistic-robotics/
- 内容：Bayes filter、sensor/motion model、belief update、定位和建图的统一概率基础。
- 相关性：直接支撑 BeliefMap、BinarySensorModel、BayesianBeliefUpdater 和显式处理感知不确定性的思想。

### R4. Optimal Eye Movement Strategies in Visual Search

- 信息：Jiri Najemnik, Wilson S. Geisler. Nature 434, 387–391, 2005. DOI: 10.1038/nature03390。
- 论文：https://www.nature.com/articles/nature03390
- 内容：ideal Bayesian visual searcher 使用场景统计、可见性和 posterior 选择最能获得目标位置信息的下一 fixation。
- 相关性：直接对应“未知目标位置 + 视点传感模型 + Bayesian 更新 + 下一观测决策”。
- 区别：该工作研究人眼/二维视觉搜索；本仓库研究 UAV/机器人空间视点。

### R5. Information-Theoretic Based Target Search with Multiple Agents

- 信息：Minkyu Kim, Ryan Gupta, Luis Sentis. arXiv:2107.12715, 2021。
- 论文：https://arxiv.org/abs/2107.12715
- 内容：从 global waypoints 和 local frontiers 生成候选路径，按信息增益选路，支持异构多机器人目标搜索和真机验证。
- 相关性：与 ActiveSearchPolicy 的候选选择和 expected information gain 非常接近，也是未来多 UAV 扩展的重要对比。
- 区别：当前 SearchSession 没有实现该文的多机器人序贯协调。

### R6. A Mathematical Theory of Communication

- 信息：Claude E. Shannon. Bell System Technical Journal 27, 1948, 379–423、623–656。
- 论文：https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
- 内容：信息熵、条件熵和互信息的数学基础。
- 相关性：直接对应 entropy_nats、binary entropy、information_gain_nats 和 utility 中 mutual information 项。

### R7. On Information and Sufficiency

- 信息：Solomon Kullback, Richard A. Leibler. Annals of Mathematical Statistics 22(1), 79–86, 1951. DOI: 10.1214/aoms/1177729694。
- 论文：https://doi.org/10.1214/aoms/1177729694
- 内容：提出 information divergence。
- 相关性：直接对应 KL divergence，用于量化观测前后 belief 的改变。

### R8. Coverage of Known Spaces: The Boustrophedon Cellular Decomposition

- 信息：Howie Choset. Autonomous Robots 9, 247–253, 2000. DOI: 10.1023/A:1008958800904。
- 论文：https://doi.org/10.1023/A%3A1008958800904
- 内容：将已知空间分解为 cells，并在 cell 中用往返运动完成覆盖。
- 相关性：支撑 CoveragePolicy 的 lawnmower/back-and-forth baseline。
- 边界：仓库不是完整 boustrophedon decomposition 复现，应称为经典 CPP 思路的几何实现。

### R9. Coverage for Robotics – A Survey of Recent Results

- 信息：Howie Choset. Annals of Mathematics and Artificial Intelligence 31, 113–126, 2001. DOI: 10.1023/A:1016639210559。
- 论文：https://doi.org/10.1023/A%3A1016639210559
- 内容：系统总结 heuristic、randomized 和 cellular-decomposition coverage planning。
- 相关性：用于定位 Coverage baseline，并区分“几何全覆盖”和“目标条件化主动搜索”。

### R10. On Evaluation of Embodied Navigation Agents

- 信息：Peter Anderson et al. arXiv:1807.06757, 2018。
- 论文：https://arxiv.org/abs/1807.06757
- 内容：统一 embodied navigation 的任务与评测方法，提出 Success weighted by Path Length（SPL）。
- 相关性：直接对应仓库的 SPL 实现。
- 注意：仓库把 shortest path 具体化为“初始点到可观察真实目标的最近候选视点距离”，论文应称为 search-adapted SPL 并给出公式。

## 3. 高度相关的延伸工作

### R11. Active Object Tracking using Context Estimation

- 信息：Minkyu Kim, Luis Sentis. arXiv:1912.06754, 2019。
- 论文：https://arxiv.org/abs/1912.06754
- 内容：用 contextual information 和 Dynamic Bayesian Network 估计目标状态，以信息论 utility 选动作，并用 POMDP 表述高层决策。
- 相关性：同样结合上下文先验、posterior 和信息收益；可用来说明本仓库当前只处理单静态目标和离散 cell belief。

### R12. Coordinated Search for a Lost Target in a Bayesian World

- 信息：Timothy Chung, Joel Burdick. Advanced Robotics 18(10), 2004。
- 论文：https://www.tandfonline.com/doi/abs/10.1163/1568553042674707
- 内容：多个传感平台维护单个非逃逸目标的 Bayesian PDF，并进行去中心化协调搜索。
- 相关性：为把当前单 session 扩展成多机器人共享 belief/协同视点提供经典参考。

### R13. Planning and Acting in Partially Observable Stochastic Domains

- 信息：Leslie Pack Kaelbling, Michael L. Littman, Anthony R. Cassandra. Artificial Intelligence 101(1–2), 99–134, 1998. DOI: 10.1016/S0004-3702(98)00023-X。
- 论文：https://doi.org/10.1016/S0004-3702(98)00023-X
- 内容：POMDP belief-state planning 基础。
- 相关性：目标搜索自然可表述为 POMDP；但当前 ActiveSearchPolicy 是 one-step myopic utility，并没有求完整 value function。

### R14. FUEL: Fast UAV Exploration Using Incremental Frontier Structure and Hierarchical Planning

- 信息：Boyu Zhou, Yichen Zhang, Xinyi Chen, Shaojie Shen. IEEE RA-L 6(2), 779–786, 2021. DOI: 10.1109/LRA.2021.3051563。
- 论文：https://doi.org/10.1109/LRA.2021.3051563
- 开源：https://github.com/HKUST-Aerial-Robotics/FUEL
- 内容：增量 frontier structure、全局覆盖、局部视点精炼和 minimum-time trajectory 的分层 UAV 探索。
- 相关性：对应 UAV candidate viewpoint、高层搜索/底层轨迹解耦和真实飞行验证。
- 区别：本仓库搜索已知几何区域中的目标，不是 frontier-based 3D mapping，也没有 minimum-time trajectory optimization。

### R15. Frontier-Based Exploration Using Multiple Robots

- 信息：Brian Yamauchi. Autonomous Agents 1998, 47–53. DOI: 10.1145/280765.280773。
- 论文：https://doi.org/10.1145/280765.280773
- 内容：把已知自由空间和未知空间的边界作为 exploration goals，并扩展到多机器人。
- 相关性：可作为候选目标生成经典对比；本仓库从已知 SearchGrid cells 生成候选，不是 frontier exploration。

## 4. 开放词汇感知参考

### R16. YOLO-World: Real-Time Open-Vocabulary Object Detection

- 信息：Tianheng Cheng et al. CVPR 2024, 16901–16911。
- 论文：https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html
- 开源：https://github.com/AILab-CVC/YOLO-World
- 内容：通过 vision-language pretraining 把 YOLO 扩展为实时开放词汇/零样本检测器。
- 相关性：ROS bridge README 将其列为未来 detector，SearchTarget 的 query/prompts 也为它保留接口。
- 边界：目前未集成到核心实验，不能声称当前系统已实现 open-vocabulary search。

### R17. Grounding DINO

- 全名：Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection。
- 信息：Shilong Liu et al. ECCV 2024. arXiv:2303.05499。
- 论文：https://arxiv.org/abs/2303.05499
- 开源：https://github.com/IDEA-Research/GroundingDINO
- 内容：结合 language grounding 和 transformer detector，通过自由文本定位开放集目标。
- 相关性：同样被 ROS bridge README 列为未来 detector，与自然语言 target query 直接匹配。

### R18. Semantic Mapping（SemExp/PONI）

- 内容：由 RGB-D、pose 和第一人称语义分割，通过几何投影形成 allocentric top-down semantic map。
- 相关性：与 SearchObservationAdapter 把 point cloud/visible ground points 投影到 SearchGrid 相关。
- 区别：当前 SemanticGridBuilder 主要从已知 scene graph 标注 grid，不是在线从 RGB-D 建完整地图。论文必须公开这一环境知识假设。

## 5. 代码思想到 reference 的逐项映射

| 仓库思想 | 建议 reference | 关系 | 必须说明的边界 |
| --- | --- | --- | --- |
| categorical target belief | Thrun et al. 2005 | 直接基础 | 单目标、单 cell |
| binary positive/negative likelihood | Thrun et al. 2005 | 直接基础 | $P_D/P_{FA}$ 为简化 sensor model |
| Shannon entropy | Shannon 1948 | 直接 | 自然对数，单位 nat |
| KL divergence | Kullback & Leibler 1951 | 直接 | posterior 相对 prior |
| mutual-information viewpoint gain | Shannon 1948；Najemnik 2005；Kim 2021 | 直接+相近应用 | binary visible/not-visible MI |
| myopic NBV | active visual search/NBV | 高度相关 | 非 non-myopic trajectory planning |
| semantic task-conditioned prior | SemExp 2020；PONI 2022 | 高度相关 | 本仓库改用 LLM weights |
| confidence mixing with uniform | robust-prior engineering | 本仓库具体设计 | 需 ablation/calibration，不虚构出处 |
| novelty mass | exploration bonus/coverage gain | 概念相关 | 具体公式是本仓库设计 |
| travel penalty | informative path planning | 概念相关 | 当前是直线距离 |
| lawnmower baseline | Choset 2000/2001 | 直接类别 | 非完整 exact decomposition |
| multi-observation confirmation | sequential detection 思路 | 工程相关 | 当前是阈值规则，非 SPRT/JPDA/RFS |
| SPL | Anderson et al. 2018 | 直接改编 | 公开 search-specific 定义 |
| POMDP | Kaelbling et al. 1998 | 理论上位 | 当前不是 POMDP solver |

## 6. 可复用或对比的开源项目

| 项目 | 地址 | 用途 |
| --- | --- | --- |
| PONI | https://github.com/srama2512/PONI | semantic ObjectNav 主要对比 |
| SemExp | https://github.com/devendrachaplot/Object-Goal-Navigation | 语义地图和目标条件化探索 |
| FUEL | https://github.com/HKUST-Aerial-Robotics/FUEL | 3D UAV frontier/NBV/trajectory |
| Habitat-Lab | https://github.com/facebookresearch/habitat-lab | 标准 ObjectNav 环境与评测 |
| YOLO-World | https://github.com/AILab-CVC/YOLO-World | 开放词汇感知后端 |
| Grounding DINO | https://github.com/IDEA-Research/GroundingDINO | 文本条件化检测 |
| PX4-Autopilot | https://github.com/PX4/PX4-Autopilot | UAV SITL/飞控 |
| MAVROS | https://github.com/mavlink/mavros | ROS–MAVLink 接口 |
| Gazebo | https://github.com/gazebosim/gz-sim | 物理与传感仿真 |

正式论文应优先引用这些项目的官方论文、Zenodo DOI 或 software citation，而不是只引用 GitHub URL。

## 7. 相关专利与 prior art

> 以下仅用于了解 prior art，不表示本项目采用或侵犯相关权利要求，不构成法律意见或 freedom-to-operate 结论。

### P1. CN113505646A/B — Target searching method based on semantic map

- 申请人：清华大学；优先权日 2021-06-10；已授权版本 CN113505646B。
- 链接：https://patents.google.com/patent/CN113505646A/en
- 内容：建立目标与父类物体语义关系图；从多视角 RGB-D 做 3D 语义重建；未知目标时访问相关父类物体附近导航点并更新关系图。
- 相关性：语义地图、物体共现关系、搜索点选择。
- 区别：本仓库是 LLM label weights + Bayesian/NBV，并非关系图递归搜索。

### P2. CN119223305A — Multi-robot visual semantic navigation method

- 链接：https://patents.google.com/patent/CN119223305A/en
- 内容：多机器人通信完善语义地图；规划综合目标存在概率和位置预测不确定性，持续选择下一目标位置。
- 相关性：与 belief、探索利用权衡和 online replanning 很接近；还覆盖本仓库未实现的多机器人地图共享。

### P3. US9696430B2 — Method and apparatus for locating a target using an autonomous UAV

- 链接：https://patents.google.com/patent/US9696430B2/en
- 内容：UAV 在指定兴趣区内寻找先验位置未知的目标。
- 相关性：应用场景相近，但不是 Bayesian semantic active search 的直接方法来源。

### P4. CN120162394B — Multimodal large model and Graph RAG semantic mapping

- 链接：https://patents.google.com/patent/CN120162394B/en
- 内容：用多模态大模型、语义数据库、object co-occurrence graph 和 Graph RAG 增强机器人语义地图。
- 相关性：与“用大模型把任务/环境语义转换为空间知识”相近。
- 区别：当前仓库没有 Graph RAG 或多模态在线建图。

## 8. 应作为本项目具体设计来描述的部分

1. SearchTask → SearchState → SearchObservation → SearchOutcome 的平台无关契约；
2. LLM label weights 投影到 grid，再与 uniform 做 confidence mixing；
3. detection probability + binary MI + quality-weighted novelty − normalized travel cost 的 utility；
4. localized/unlocalized positive evidence 与 negative evidence 的统一；
5. detection confidence × observation quality 修正有效检测率；
6. 依据确认次数、持续时间和定位误差进入 verification mode；
7. simulator ground truth 隔离在 observation interface 后；
8. 同一 SearchSession 复用于 synthetic、semantic simulator 和 ROS/Gazebo。

这些组合需要实验论证，不能仅凭模块组合就主张创新。

## 9. 建议的 Related Work 结构

1. **Semantic Object-Goal Navigation**：SemExp、PONI；说明室内 ObjectNav 与室外 UAV search、learned potential 与 LLM prior 的差别。
2. **Bayesian and Information-Theoretic Active Search**：Bayes filter、active visual search、information gain；突出 LLM prior、negative evidence、sensor quality 和 verification。
3. **Coverage, Frontier and UAV Exploration**：CPP、Yamauchi、FUEL；解释 coverage baseline 和目标条件化 active sensing 的差异。
4. **Open-Vocabulary Perception**：YOLO-World、Grounding DINO；若实验仍用颜色 detector，只能写成未来感知后端。

## 10. 距离合格论文仍需补充的工作

### A. 固定问题与创新点

1. 聚焦“LLM task-conditioned prior + Bayesian active UAV search”，不要把整个 GSI 都列作主贡献。
2. 写出 2–4 条可验证 contribution，每条对应方法和实验。
3. 明确区别于 PONI、SemExp 和经典 information-theoretic search。

### B. 方法完整性

1. 正式定义 state、action、observation、likelihood、budget、termination 和 objective。
2. 给出完整伪代码。
3. 解释 utility 各项尺度、权重和调参方式。
4. 校准或实证论证 $P_D/P_{FA}$、observation quality 和 confidence。
5. 加入 camera frustum、ray casting、occlusion-aware visibility。
6. 把直线 travel cost 升级为可行路径长度或飞行时间。

### C. 必需基线

1. Coverage、Random（已有）；
2. greedy current-posterior mass；
3. information-gain-only NBV；
4. frontier baseline（未知地图设置）；
5. PONI/SemExp-style semantic baseline 或可比任务对照；
6. oracle prior、uniform prior；
7. 若声称开放词汇，加入 YOLO-World/Grounding DINO。

### D. 必需 Ablation

1. 移除 LLM prior；
2. 移除 Bayesian negative evidence；
3. 分别移除 detection、IG、novelty、travel；
4. 移除 verification mode；
5. confidence mixing 对比直接归一化 LLM weights；
6. localized/unlocalized/negative evidence 分测；
7. grid resolution、stride、altitude、FOV、budget 敏感性。

### E. 数据和实验规模

1. 多地图、多目标类别、多目标位置、多 prior condition；
2. held-out maps、categories、language instructions；
3. 分开报告 synthetic、Gazebo/PX4 SITL、真实机器人；
4. 真机覆盖不同光照、高度、遮挡、传感延迟；
5. 分类报告误报、漏检、错误先验、定位失败、控制失败、预算耗尽。

### F. 单独评价 LLM Prior

1. target-cell NLL、multi-class Brier score；
2. target-cell rank、top-k recall、calibration curve/ECE；
3. 不同 LLM、prompt、temperature、label inventory 的稳定性；
4. token、延迟、成本；
5. 错误/敌意 prior 下 Bayesian negative evidence 的恢复速度。

### G. 指标和统计

1. success、false-positive、false-negative、search-adapted SPL；
2. time/distance/energy-to-detection；
3. posterior NLL/Brier/entropy、localization error；
4. paired bootstrap confidence intervals；
5. success 用 paired McNemar test；
6. 连续指标用 paired Wilcoxon，或验证前提后用 paired t-test；
7. 多比较用 Holm correction，并报告 effect size；
8. 小样本 Bernoulli rate 用 Wilson/Clopper–Pearson 或 paired bootstrap，不只用 normal CI。

### H. 可复现性

1. 冻结 commit、Docker image、依赖版本、seeds；
2. 发布 configs、场景生成器、raw traces、聚合脚本；
3. 一键复现主表与 ablation；
4. 严格拆分训练/调参/测试地图；
5. 公开计算资源、运行时长、LLM API、失败重试规则；
6. 提供数据、模型、地图和第三方代码 license/provenance 表。

## 11. 建议图表

1. Figure 1：task → semantic prior → belief → viewpoint → observation → posterior。
2. Figure 2：单 episode 的 prior/posterior heatmap、视点和检测演化。
3. Figure 3：正确、均匀、误导 prior 下的行为。
4. Table 1：Coverage/Random/Greedy/IG-only/semantic baseline/Full method。
5. Table 2：prior、utility、update、verification ablation。
6. Table 3：跨地图、目标、压力条件泛化。
7. Table 4：synthetic、Gazebo/PX4、real-world sim-to-real gap。
8. Appendix：推导、超参数、prompt、场景、统计、failure cases。

## 12. 建议的核心叙事

不宜声称“首次使用 Bayesian search”或“首次使用 information-gain NBV”。更稳妥的叙事是：

> 现有 semantic ObjectNav 通常在室内环境学习 long-term goal 或 potential，经典 Bayesian target search 又往往假定 prior 和 sensor model 已给定。本文研究如何把自然语言任务与公开语义地图转换为可审计的空间先验，并在开放室外 UAV 搜索中用真实观测持续纠正它，同时统一处理信息收益、运动代价、传感质量和多次确认。

这一叙事仍需强基线、完整 ablation、held-out 多场景、严格统计和真机结果支撑。

## 13. 优先级

### 投稿前必做

1. 冻结主张和 contributions；
2. 补齐上述核心 Related Work；
3. 完成核心 ablation；
4. 增加强语义/NBV 基线；
5. 建立 held-out multi-map benchmark；
6. 单独评价 LLM prior calibration；
7. 用 paired statistics 重做结果；
8. 完成足量 Gazebo/PX4 实验，最好补真机。

### 后续扩展

1. 真正 open-vocabulary detector；
2. occlusion-aware 3D viewpoint generation；
3. non-myopic POMDP/informative path planning；
4. 动态目标与多目标 belief；
5. 多 UAV 共享 belief 和协同分配；
6. 学习 utility weights 或端到端 policy。


