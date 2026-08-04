# Horizon-2 Belief Lookahead 定义

## 目标

`lookahead_active` 在现有 active-search 即时效用之上，显式计算一次未来观测后的最优下一步。它用于回答：当前 viewpoint 是否会产生一个更有价值的后续决策状态。

## 状态与动作

决策状态为：

\[
s_t=(b_t, V_t, q_t, c_t, B_t)
\]

其中 `b_t` 是目标在搜索单元上的 categorical belief，`V_t` 是未访问 viewpoint，`q_t` 是观测质量，`c_t` 是当前位置，`B_t` 是已消耗预算。动作 `v` 是一个未访问 viewpoint。

## 二值观测模型

令 `F(v)` 为 viewpoint 的可见单元集合，`p_d(q)` 为有效检测率，`p_fp` 为假阳性率。对于目标单元 `x`：

\[
P(z=1\mid x,v)=
\begin{cases}
p_d(q), & x\in F(v)\\
p_{fp}, & x\notin F(v)
\end{cases}
\]

分支概率通过 belief 边缘化：

\[
P(z\mid b,v)=\sum_x b(x)P(z\mid x,v)
\]

## Belief transition

每个观测分支都使用 Bayes update：

\[
\tau(b,v,z)(x)=\frac{b(x)P(z\mid x,v)}
{\sum_{x'}b(x')P(z\mid x',v)}
\]

该 likelihood 与 `BinarySensorModel` 和 `BayesianBeliefUpdater` 的 unlocalized positive/negative evidence 一致。

## Reward 与 Horizon-2 value

即时 reward 沿用 `ActiveSearchPolicy` 的可审计效用：

\[
r(b,v)=w_dP(z=1\mid b,v)+w_iIG(b,v)+w_nN(b,v)-w_cC(v)
\]

Horizon-2 action value 为：

\[
Q_2(b,v)=r(b,v)+\gamma\sum_{z\in\{0,1\}}P(z\mid b,v)
\max_{v'\in V\setminus\{v\}}r(\tau(b,v,z),v')
\]

策略选择 `argmax_v Q_2(b,v)`。`gamma=0` 必须退化为现有 greedy active policy，这也是实现的回归约束。

## 近似与审计输出

当前实现只展开两层和两个观测分支，复杂度为 `O(2|V|^2)`。每次决策记录 immediate score、分支概率、posterior entropy、每个分支的最优 continuation candidate/value，以及最终 `Q_2`，可直接用于后续 teacher dataset。

## 当前边界

- 假想 transition 更新 belief、覆盖、当前位置、距离与 step；不估算平台相关的时间和能量。
- positive branch 使用 unlocalized positive likelihood；带定位误差的细粒度 positive observation branching 留给 3D/realism 阶段。
- reward 仍是复合 surrogate。与“预算内真实发现概率”之间的相关性必须通过 paired benchmark 验证，不能仅依赖 utility 提升下结论。
