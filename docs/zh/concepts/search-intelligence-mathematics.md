# GSI Search Intelligence 数学定义与 Benchmark 规范

本文给出当前 GSI Search Intelligence 实现所对应的完整数学定义。公式与
`modules/search_intelligence/` 中的代码逐项对应；尚未实现的研究扩展会明确标注，
不会与当前方法混写。

## 1. 问题定义

给定自然语言任务 $T$、开放室外搜索区域 $\mathcal A$、语义地图 $M$、机器人初始状态
$s_0$、传感器模型 $\Theta$ 和资源预算 $B$，搜索策略需要依次选择观测视点

$$
v_t=(x_t,y_t,z_t,\psi_t,\theta_t)\in\mathcal V,
$$

并在每次观测后更新目标位置 belief，直到发现目标或预算耗尽。研究目标不是低层飞行控制，
而是学习或构造一个 task-conditioned 决策规则

$$
\pi(v_t\mid T,M,b_t,s_t),
$$

使目标发现概率、搜索效率和信息收益最大，同时控制时间、距离和能量代价。

当前实现采用离散目标位置假设：一个目标位于一个可搜索网格单元中。设可搜索单元集合为

$$
\mathcal C=\{c_1,\ldots,c_N\},\qquad X\in\mathcal C,
$$

其中 $X$ 是未知目标位置。时刻 $t$ 的 belief 为

$$
b_t(i)=P(X=c_i\mid z_{1:t},v_{1:t},T,M),
\qquad
b_t(i)\ge 0,
\qquad
\sum_{i=1}^{N}b_t(i)=1.
$$

## 2. 符号表

| 符号 | 含义 | 对应实现 |
| --- | --- | --- |
| $T$ | 自然语言条件化搜索任务 | `SearchTask` |
| $M$ | 环境语义地图 | `SemanticGridBuilder` |
| $\mathcal C$ | 可搜索网格单元集合 | `SearchGrid.searchable_cells` |
| $b_t(i)$ | 目标在单元 $c_i$ 的 posterior probability | `SearchState.belief` |
| $\mathcal V$ | 候选视点集合 | `CandidateViewpointGenerator` |
| $V(v)$ | 视点 $v$ 可见的网格单元集合 | `visible_cell_ids` |
| $z_t\in\{0,1\}$ | 第 $t$ 次观测是否产生有效目标 detection | `SearchObservation` |
| $P_D$ | 目标可见时的 detection probability | `detection_probability` |
| $P_{FA}$ | 目标不可见时的 false-positive probability | `false_positive_probability` |
| $q_t$ | 观测质量，范围 $[0,1]$ | `observation_quality` |
| $o_t(i)$ | 单元 $c_i$ 历史最高观测质量 | `observed_cell_quality` |
| $d(v_t,v)$ | 当前视点到候选视点的三维欧氏距离 | `_travel_distance` |
| $H(b)$ | belief 的 Shannon entropy，单位 nat | `entropy_nats` |
| $I(X;Z_v)$ | 候选视点的期望信息增益 | `information_gain_nats` |

## 3. 搜索空间与候选视点

### 3.1 网格离散化

搜索区域的 axis-aligned bounding box 为
$[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]$，网格分辨率为 $r$。网格尺寸为

$$
W=\max\left(1,\left\lceil\frac{x_{\max}-x_{\min}}{r}\right\rceil\right),
\qquad
H=\max\left(1,\left\lceil\frac{y_{\max}-y_{\min}}{r}\right\rceil\right).
$$

单元 $c_{jk}$ 的中心为

$$
\bar x_{jk}=\frac{x_{jk}^{\min}+x_{jk}^{\max}}{2},
\qquad
\bar y_{jk}=\frac{y_{jk}^{\min}+y_{jk}^{\max}}{2}.
$$

只有中心位于搜索几何内部、且不属于 restricted geometry 的单元进入 $\mathcal C$。

### 3.2 候选视点

对满足 stride 条件的可搜索单元

$$
j\bmod s=0,\qquad k\bmod s=0,
$$

在其中心上方生成 nadir-camera 候选视点

$$
v_{jk}=(\bar x_{jk},\bar y_{jk},h,\psi,-\pi/2).
$$

若直接配置 footprint radius $R_f$，则使用该值；否则由飞行高度和水平 FOV $\phi$ 得到

$$
R_f=h\tan\left(\frac{\phi}{2}\right).
$$

候选视点的可见单元集合为

$$
V(v)=\{c_i\in\mathcal C:
\sqrt{(\bar x_i-x_v)^2+(\bar y_i-y_v)^2}\le R_f\}.
$$

当前候选生成器使用圆形地面 footprint；真实相机标定、遮挡和 terrain occlusion 尚未进入该公式。

## 4. LLM Task-Conditioned 语义先验

### 4.1 LLM 的输入与输出

LLM 输入不是目标坐标，而是

$$
R_T=(\text{target query},\text{attributes},\text{instruction},
\text{semantic inventory},\text{context priors},\text{excluded regions}).
$$

LLM 输出经 `SearchPrior` 验证后包含：语义权重 $w_\ell\ge 0$、整体 confidence
$\alpha\in[0,1]$、default weight $w_0\ge 0$ 和排除标签集合 $\mathcal E$。

必须强调：$w_\ell$ 是 LLM 给出的相对语义相关性分数，不是经过证明或校准的 Bayesian
probability。只有经过投影、归一化和 confidence mixing 后，结果才成为合法的空间 prior。

### 4.2 单元 raw score

设单元 $c_i$ 的语义标签集合为 $L_i$。当前实现使用最大匹配权重：

$$
r_i=
\begin{cases}
0, & L_i\cap\mathcal E\ne\varnothing,\\[4pt]
\max\left(\{w_0\}\cup\{w_\ell:\ell\in L_i\}\right),
& L_i\cap\mathcal E=\varnothing.
\end{cases}
$$

设未被排除的 eligible cell 集合为 $\mathcal C_E$。若全部单元被错误排除，代码回退到
$\mathcal C_E=\mathcal C$。

### 4.3 语义分布与 confidence mixing

若 $\sum_{c_i\in\mathcal C_E}r_i>0$，语义分布为

$$
p_{\mathrm{sem}}(i)=
\frac{r_i}{\sum_{c_j\in\mathcal C_E}r_j}.
$$

若 raw score 总和为零，则回退到 eligible cells 上的 uniform distribution：

$$
p_{\mathrm{sem}}(i)=\frac{1}{|\mathcal C_E|}.
$$

最终初始 belief 是 semantic distribution 与 uniform uncertainty 的 convex mixture：

$$
b_0(i)=
\alpha p_{\mathrm{sem}}(i)
+(1-\alpha)\frac{1}{|\mathcal C_E|},
\qquad c_i\in\mathcal C_E,
$$

排除单元的 $b_0(i)=0$。因此

$$
b_0(i)\ge0,
\qquad
\sum_i b_0(i)=1.
$$

当没有 LLM prior 时，系统使用

$$
b_0(i)=\frac{1}{N}.
$$

## 5. Binary Sensor Model

设基础 sensor parameters 为

$$
P_D=P(z=1\mid X\in V(v)),
\qquad
P_{FA}=P(z=1\mid X\notin V(v)),
\qquad
P_D>P_{FA}.
$$

观测质量 $q\in[0,1]$ 将 detector 从基础性能插值到 uninformative sensor：

$$
\widetilde P_D(q)
=P_{FA}+q(P_D-P_{FA}).
$$

于是 $q=1$ 时 $\widetilde P_D=P_D$，$q=0$ 时
$\widetilde P_D=P_{FA}$，即观测不能区分可见和不可见目标。

若存在超过任务 confidence threshold $\tau$ 的 detection，令最高 detection confidence 为
$c_{\max}$，正证据使用的有效质量为 $q^+=q c_{\max}$：

$$
\widetilde P_D^+=P_{FA}+qc_{\max}(P_D-P_{FA}).
$$

## 6. Bayesian Belief Update

统一更新式为

$$
b_{t+1}(i)
=\eta_t L_t(i)b_t(i)
=\frac{L_t(i)b_t(i)}{\sum_{j=1}^{N}L_t(j)b_t(j)},
$$

其中 $L_t(i)=P(z_t\mid X=c_i,v_t)$，$\eta_t$ 是 normalization constant。
为避免数值下溢，代码实际使用

$$
L_t(i)\leftarrow\max(\epsilon,L_t(i)),
\qquad \epsilon=10^{-9}\ \text{(default)}.
$$

### 6.1 Localized positive evidence

若 detection 可以投影到一个或多个网格单元，记定位后的 evidence set 为
$E_t\subseteq\mathcal C$：

$$
L_t^+(i)=
\begin{cases}
\widetilde P_D^+, & c_i\in E_t,\\
P_{FA}, & c_i\notin E_t.
\end{cases}
$$

### 6.2 Unlocalized positive evidence

若 detection 有效但没有 estimated position，则使用当前视点可见集合
$E_t=V(v_t)$，并采用同一个正证据 likelihood。它会把 probability mass 推向可见区域，
但不能在可见单元内部进一步定位。

### 6.3 Negative evidence

若没有超过阈值 $\tau$ 的 detection，则

$$
L_t^-(i)=
\begin{cases}
1-\widetilde P_D(q_t), & c_i\in V(v_t),\\
1-P_{FA}, & c_i\notin V(v_t).
\end{cases}
$$

因此高质量的 negative observation 会降低可见区域的 posterior mass；零质量观测满足
$\widetilde P_D=P_{FA}$，归一化后保持 $b_{t+1}=b_t$。

### 6.4 两单元示例

若 $b_t=(0.5,0.5)$，只看见 $c_1$，且 $P_D=0.8$、$P_{FA}=0.1$，没有检测到目标，则

$$
\tilde b_{t+1}=(0.5(1-0.8),\ 0.5(1-0.1))=(0.1,0.45),
$$

归一化得到

$$
b_{t+1}=\left(\frac{2}{11},\frac{9}{11}\right).
$$

若反而在 $c_1$ 得到 confidence $1$ 的 localized positive detection，则

$$
\tilde b_{t+1}=(0.5\times0.8,\ 0.5\times0.1)=(0.4,0.05),
$$

$$
b_{t+1}=\left(\frac{8}{9},\frac{1}{9}\right).
$$

## 7. Belief Diagnostics

### 7.1 Entropy

$$
H(b_t)=-\sum_{i:b_t(i)>0}b_t(i)\ln b_t(i).
$$

单位是 nat。等效候选单元数量定义为 perplexity：

$$
N_{\mathrm{eff}}(b_t)=\exp(H(b_t)).
$$

单次实际 entropy reduction 为

$$
\Delta H_t=H(b_t)-H(b_{t+1}).
$$

注意 $\Delta H_t$ 可以为负，因为某次观测可能让 posterior 更分散；代码不会强制截断。

### 7.2 KL divergence

当前更新记录 posterior 相对 prior 的 information change：

$$
D_{\mathrm{KL}}(b_{t+1}\Vert b_t)
=\sum_{i:b_{t+1}(i)>0}
b_{t+1}(i)\ln\frac{b_{t+1}(i)}{b_t(i)}.
$$

若 prior probability 为零，该项在实现中跳过。正常 Bayesian update 不会从零 prior
产生正 posterior，因此该处理与当前模型一致。

## 8. Policy 定义

所有 policy 都排除已经访问过的 viewpoint，并受剩余 viewpoint budget 限制。若当前预算已耗尽，
`select_next` 返回空并终止。

### 8.1 CoveragePolicy

Coverage baseline 不读取 task semantics 和 belief。对 polygon/rectangle，沿 $y$ 方向生成
lawnmower scanlines。设区域高度为 $L_y=y_{\max}-y_{\min}$，pass spacing 为 $s_p$：

$$
n_p=\max\left(2,\left\lceil\frac{L_y}{s_p}\right\rceil\right),
$$

$$
y_k=y_{\min}+\frac{k}{n_p-1}(y_{\max}-y_{\min}),
\qquad k=0,\ldots,n_p-1.
$$

每条 scanline 与 polygon 边界相交产生左右端点 $(x_k^L,y_k)$、$(x_k^R,y_k)$；偶数行从左到右，
奇数行从右到左。若 observation spacing 为 $s_o$，长度为 $d$ 的 route segment 被采样为

$$
n_s=\max\left(1,\left\lceil\frac{d}{s_o}\right\rceil\right)
$$

个等距子段。每个 viewpoint 的 yaw 由下一点方向得到：

$$
\psi_k=\mathrm{atan2}(y_{k+1}-y_k,x_{k+1}-x_k).
$$

实现还支持 circle chord coverage、polyline band coverage、line back-and-forth 和 point spiral；
这些只是不同 search geometry 的 deterministic coverage route，决策原则仍然是几何覆盖。

### 8.2 RandomPolicy

Random baseline 对每个未访问候选 $v$ 计算稳定 hash key：

$$
k(v)=\mathrm{SHA256}(\mathrm{seed}:\mathrm{candidateId}(v)),
$$

然后按 $k(v)$ 的 byte order 排序。它是 seeded、可复现的随机排列，不读取 belief。

### 8.3 GreedyPriorPolicy

GreedyPrior 使用固定初始先验 $b_0$，不会使用后续 Bayesian posterior。候选覆盖的 prior mass 为

$$
m_0(v)=\sum_{c_i\in V(v)}b_0(i).
$$

其 utility 为

$$
U_{\mathrm{greedy}}(v)
=m_0(v)-\lambda_d^{G}\frac{d(v_t,v)}{d_0}.
$$

当前 benchmark 中 $\lambda_d^{G}=0$，所以它纯粹按 initial prior mass 排序。utility 相同时依次按
$m_0(v)$ 和 `candidate_id` 确定顺序。

### 8.4 ActiveSearchPolicy

Active policy 每一步都使用最新 posterior $b_t$ 重新评分。

#### 可见 belief mass

$$
p_v=P(X\in V(v)\mid b_t)
=\sum_{c_i\in V(v)}b_t(i).
$$

#### 预测 detection probability

令候选观测的二值输出为 $Z_v$，则

$$
q_v=P(Z_v=1)
=p_vP_D+(1-p_v)P_{FA}.
$$

代码中的 `detection_probability` 就是 $q_v$。

#### 期望信息增益推导

定义二元变量 $Y_v=\mathbb 1[X\in V(v)]$。Binary entropy 为

$$
h(p)=-p\ln p-(1-p)\ln(1-p),
$$

其中 $h(0)=h(1)=0$。候选观测的信息增益是 mutual information：

$$
\mathrm{IG}(v)=I(Y_v;Z_v)
=H(Z_v)-H(Z_v\mid Y_v).
$$

因为 $P(Z_v=1)=q_v$，且
$P(Z_v=1\mid Y_v=1)=P_D$、
$P(Z_v=1\mid Y_v=0)=P_{FA}$，所以

$$
\boxed{
\mathrm{IG}(v)
=h(q_v)-p_vh(P_D)-(1-p_v)h(P_{FA})
}
$$

单位为 nat。实现对浮点误差使用
$\mathrm{IG}(v)\leftarrow\max(0,\mathrm{IG}(v))$。

#### Novelty

单元历史观测质量为

$$
o_t(i)=\max_{\tau\le t:\,c_i\in V(v_\tau)}q_\tau,
$$

未观测单元令 $o_t(i)=0$。候选 novelty mass 为

$$
N_t(v)=\sum_{c_i\in V(v)}b_t(i)(1-o_t(i)).
$$

它让 policy 优先观察具有较高 posterior mass、但尚未被高质量观测覆盖的单元。

#### Travel cost

$$
d(v_t,v)=\sqrt{(x_t-x_v)^2+(y_t-y_v)^2+(z_t-z_v)^2},
$$

$$
C_d(v)=\frac{d(v_t,v)}{d_0}.
$$

#### 最终 utility

$$
\boxed{
U_{\mathrm{active}}(v)=
\lambda_{\mathrm{det}}q_v
+\lambda_{\mathrm{IG}}\mathrm{IG}(v)
+\lambda_{\mathrm{nov}}N_t(v)
-\lambda_{\mathrm{travel}}C_d(v)
}
$$

下一视点为

$$
v_{t+1}=\arg\max_{v\in\mathcal V\setminus\mathcal V_{\mathrm{visited}}}
U_{\mathrm{active}}(v).
$$

utility 相同时使用 `candidate_id` 作 deterministic tie-break。若配置
$U_{\min}$，则只保留 $U(v)\ge U_{\min}$ 的候选；当前 benchmark 未配置该阈值。

#### Confirmation-aware verification

当任务要求多次确认或最小持续时间时，一次满足置信度阈值的检测不能立即声明成功。
对同一实体 $e$ 的有效检测集合记为 $\mathcal D_e^t$，它仍待复核的条件为

$$
|\mathcal D_e^t|<n_{\min}
\quad\lor\quad
\max_{d\in\mathcal D_e^t}t_d-
\min_{d\in\mathcal D_e^t}t_d<t_{\mathrm{persist}}.
$$

若最近一次待复核检测 $d_e^*$ 定位到单元 $c_e^*$，它之后覆盖该单元的 follow-up
observation 数量记为

$$
k_e^t=\sum_{\tau>t(d_e^*)}\mathbf 1[c_e^*\in V(v_\tau)].
$$

当未配置上限，或 $k_e^t<k_{\max}$ 时，可用于复核的未访问候选集合为

$$
\mathcal V_{\mathrm{verify}}(e)=
\left\{
v\in\mathcal V\setminus\mathcal V_{\mathrm{visited}}
:c_e^*\in V(v)
\right\}.
$$

只要该集合非空，policy 进入 verification mode 并选择最近的复核视点：

$$
v_{t+1}=
\arg\min_{v\in\mathcal V_{\mathrm{verify}}(e)}d(v_t,v).
$$

距离相同时使用 `candidate_id` 作 deterministic tie-break。若任务只要求一次确认，
没有可用的 `localized_cell_id`，或复核集合为空，则回退到上述
$U_{\mathrm{active}}$ 排序。这个机制只读取任务契约和 observation history；
benchmark 的 synthetic sensor 与真实仿真 sensor adapter 都应通过 detection attributes
提供观测得到的 `localized_cell_id`，policy 不读取 ground truth target cell。
`verification_followup_limit` 对应 $k_{\max}$；设为 `1` 时每次 detection 最多强制触发
一次 follow-up，未配置时持续复核到确认或没有可用候选。前者限制 false alarm 的预算消耗，
后者更积极地利用已定位 detection，两者应作为独立实验条件报告。

## 9. 资源累计、成功与终止条件

每一步的运动距离、时间和能量为

$$
d_t=d(v_{t-1},v_t),
$$

$$
\Delta t_t=\frac{d_t}{v_{\mathrm{speed}}}+t_{\mathrm{obs}},
$$

$$
\Delta E_t=e_m d_t+e_{\mathrm{obs}}.
$$

累计量为

$$
D_t=\sum_{\tau=1}^{t}d_\tau,
\qquad
T_t=\sum_{\tau=1}^{t}\Delta t_\tau,
\qquad
E_t=\sum_{\tau=1}^{t}\Delta E_\tau.
$$

当 time、distance、energy 或 viewpoint count 中任一预算首先达到上限时终止。

系统声明 FOUND 需要同一 `entity_id`（没有 ID 时使用 normalized label）的有效检测满足：

$$
c\ge\tau,
\qquad
n_{\mathrm{confirm}}\ge n_{\min},
\qquad
t_{\max}-t_{\min}\ge t_{\mathrm{persist}},
$$

若配置 localization error threshold，还需

$$
e_{\mathrm{loc}}\le e_{\max}.
$$

Benchmark 同时区分系统声明成功 `declared_found` 与 ground-truth success
`target_found`，避免把 false positive 当作真正成功。

## 10. Benchmark 实验设计

### 10.1 公平比较原则

每个 policy 在相同的 scenario、initial belief、候选视点、传感器模型、预算、起点和资源模型下运行。
对 scenario $s$、repetition $r$ 使用稳定 seed：

$$
\xi_{s,r}=\mathrm{UInt64}
\left(\mathrm{SHA256}(\text{base seed}:s:r)_{1:8}\right).
$$

一次 sensor sample 由

$$
u=\frac{\mathrm{UInt64}
(\mathrm{SHA256}(\xi_{s,r}:s:\text{viewpoint key}:\text{sensor})_{1:8})}
{2^{64}}
$$

确定。如果目标单元可见，则检测发生条件为 $u<P_D$；否则为 $u<P_{FA}$。

### 10.2 Canonical scenarios

默认区域为 $100\,\mathrm m\times80\,\mathrm m$，resolution 为 $20\,\mathrm m$，因此有
$5\times4=20$ 个 searchable cells。起点为 $(10,10,30)$，最大 viewpoint 数为 24。

| Condition | Target cell $(row,column)$ | Prior focus cells | Focus mass |
| --- | ---: | --- | ---: |
| `correct` | $(3,4)$ | $(3,4),(3,3),(2,4)$ | $0.75$ |
| `uniform` | $(3,4)$ | 无 | uniform |
| `noisy` | $(2,3)$ | $(2,3),(2,2),(1,3),(0,4)$ | $0.55$ |
| `misleading` | $(3,4)$ | $(0,0),(0,1),(1,0)$ | $0.75$ |

若 focus cell 集合为 $F$、focus mass 为 $\rho$，测试 prior 为

$$
b_0(i)=
\begin{cases}
\rho/|F|, & c_i\in F,\\[2pt]
(1-\rho)/(N-|F|), & c_i\notin F.
\end{cases}
$$

这四个条件分别检查：有用 prior、无 prior、部分噪声 prior 和错误 prior。

### 10.3 默认参数

| 参数 | 当前值 |
| --- | ---: |
| Altitude $h$ | $30\,\mathrm m$ |
| Footprint radius $R_f$ | $20\,\mathrm m$ |
| Speed $v_{\mathrm{speed}}$ | $10\,\mathrm{m/s}$ |
| Observation time $t_{\mathrm{obs}}$ | $1\,\mathrm s$ |
| Energy per meter $e_m$ | $0.05$ |
| Observation energy $e_{\mathrm{obs}}$ | $0.5$ |
| $P_D$ | $0.85$ |
| $P_{FA}$ | $0.01$ |
| Observation quality | $1.0$ |
| Detection confidence | $1.0$ |
| Coverage pass/observation spacing | $20/20\,\mathrm m$ |
| Candidate stride | $1$ cell |
| $\lambda_{\mathrm{det}}$ | $1.0$ |
| $\lambda_{\mathrm{IG}}$ | $1.0$ |
| $\lambda_{\mathrm{nov}}$ | $0.25$ |
| $\lambda_{\mathrm{travel}}$ | $0.1$ |
| Distance scale $d_0$ | $100\,\mathrm m$ |

`SearchBenchmarkConfig` 的 class default 是 10 repetitions；CLI
`run/run_search_policy_benchmark.py` 的 default 是 20 repetitions。四种 policy、四个 scenario、
$R$ 次重复共运行

$$
N_{\mathrm{episodes}}=4\times4\times R=16R.
$$

当 $R=20$ 时为 320 episodes，每种 policy 有 80 episodes。

### 10.4 Realism failure models

V5 不再假设所有 false alarm 相互独立。对固定 target-like distractor $d$，只要它位于
视点 footprint 内，就按

$$
z_t^{d}\sim\operatorname{Bernoulli}(P_d)
$$

产生 detection，并在所有视点保持相同 `entity_id`。因此它能够满足多次确认条件，直接检验
verification 是否能抵抗 persistent physical distractor。

对 correlated false alarm，先为 episode 采样 common-mode indicator

$$
C\sim\operatorname{Bernoulli}(\rho).
$$

当 $C=1$ 时，所有非目标视点共享一次

$$
A\sim\operatorname{Bernoulli}(P_{FA});
$$

当 $C=0$ 时，各视点独立采样 $A_t\sim\operatorname{Bernoulli}(P_{FA})$。
配置 shared identity 时，common-mode alarm 还共享同一 tracker ID。这是 exchangeable mixture
模型，用来分离 event correlation 和 ordinary independent false alarms。

定位误差使用 seed-deterministic 二维 Gaussian：

$$
\hat{\mathbf x}_t=\mathbf x_t+\boldsymbol\epsilon_t,
\qquad
\boldsymbol\epsilon_t\sim\mathcal N(\mathbf 0,\sigma_{loc}^{2}\mathbf I_2),
$$

$$
e_{loc,t}=\|\boldsymbol\epsilon_t\|_2.
$$

只有 $e_{loc,t}\le e_{\max}$ 的 detection 才能作为 success confirmation 和 Bayesian positive
evidence；否则该 viewpoint 作为没有有效 target detection 的 observation 更新。每次采样的
source kind、entity ID、估计位置、误差和 localized cell 都写入 `sensor_trace`。

## 11. Episode-Level Metrics

对第 $n$ 个 episode 定义 ground-truth success indicator

$$
S_n=\mathbb 1[\text{真实目标 entity 被检测到}],
$$

声明成功 indicator

$$
\widehat S_n=\mathbb 1[\text{SearchOutcome 状态为 FOUND}],
$$

false-positive indicator

$$
F_n=\mathbb 1[\widehat S_n=1\land S_n=0].
$$

### 11.1 Coverage fraction

若曾以任意正观测质量观察过的 belief cell 集合为 $\mathcal O_n$，则

$$
C_n=\frac{|\mathcal O_n\cap\mathcal C|}{|\mathcal C|}.
$$

当前指标按“访问过的 cell 数”计算，不按 physical area 或 observation quality 加权。

### 11.2 SPL

令 $L_n$ 为从初始视点直接到任一能看见目标的 candidate 的最短三维距离，$P_n$ 为实际累计距离：

$$
\mathrm{SPL}_n=
S_n\frac{L_n}{\max(L_n,P_n)}.
$$

实现的边界情况为

$$
\mathrm{SPL}_n=
\begin{cases}
0, & S_n=0,\\
1, & S_n=1,\ L_n\le0,\ P_n\le10^{-9},\\
0, & S_n=1,\ L_n\le0,\ P_n>10^{-9},\\
L_n/\max(L_n,P_n), & \text{otherwise}.
\end{cases}
$$

### 11.3 其他逐 episode 指标

系统还记录

$$
K_n=\text{viewpoint steps},\quad
T_n=\text{elapsed time},\quad
D_n=\text{distance},\quad
E_n=\text{energy},
$$

以及

$$
\Delta H_n=H(b_0)-H(b_{\mathrm{final}}).
$$

完整 belief entropy trace 为

$$
\left(H(b_0),H(b_1),\ldots,H(b_{K_n})\right).
$$

## 12. Aggregate Statistics

对某 policy/condition 下的 metric samples $x_1,\ldots,x_m$，当前代码使用 sample mean

$$
\bar x=\frac{1}{m}\sum_{n=1}^{m}x_n,
$$

sample standard deviation

$$
s=\sqrt{\frac{1}{m-1}\sum_{n=1}^{m}(x_n-\bar x)^2},
$$

以及 normal-approximation 95% confidence interval

$$
\boxed{
\bar x\pm1.96\frac{s}{\sqrt m}
}.
$$

当 $m=1$ 时 half-width 为 0。对 success rate、declared-found rate、false-positive rate、SPL
和 coverage fraction，置信区间截断到 $[0,1]$。

Success rate、declared-found rate 和 false-positive rate 都是 Bernoulli indicator 的 sample mean：

$$
\widehat{\mathrm{SR}}=\frac{1}{m}\sum_nS_n,
\qquad
\widehat{\mathrm{DR}}=\frac{1}{m}\sum_n\widehat S_n,
\qquad
\widehat{\mathrm{FPR}}=\frac{1}{m}\sum_nF_n.
$$

`successful_elapsed_time_s` 和 `successful_distance_m` 只在 $S_n=1$ 的 subset 上计算，
因此它们的 sample count 可能小于 episode count。报告同时产生：

- overall policy aggregates；
- 按 `correct/uniform/noisy/misleading` 分组的 condition aggregates；
- 每个 episode 的原始记录和 policy/belief traces。

## 13. 当前可复现的 Smoke Benchmark

使用 2026-07-30 当前代码、base seed 0、20 repetitions 运行 320 episodes，overall 结果如下。
区间为上述 normal-approximation 95% CI。

```bash
python run/run_search_policy_benchmark.py \
  --policies coverage random greedy_prior active \
  --repetitions 20 \
  --seed 0 \
  --output-dir results/search_policy_benchmark
```

该命令生成 `search_benchmark_report.json`、逐 episode 的
`search_benchmark_episodes.csv` 和 aggregate/condition statistics 的
`search_benchmark_aggregates.csv`。

| Policy | Success rate | False positive | SPL | Mean distance (m) | Successful time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coverage | $0.8625\ [0.7866,0.9384]$ | $0.1250\ [0.0521,0.1979]$ | $0.1933\ [0.1751,0.2114]$ | $316.89\ [299.11,334.67]$ | $53.31\ [52.91,53.72]$ |
| Random | $0.9250\ [0.8669,0.9831]$ | $0.0750\ [0.0169,0.1331]$ | $0.4231\ [0.3527,0.4934]$ | $278.88\ [234.29,323.48]$ | $33.83\ [28.17,39.50]$ |
| GreedyPrior | $0.9500\ [0.9019,0.9981]$ | $0.0500\ [0.0019,0.0981]$ | $0.4573\ [0.3871,0.5275]$ | $292.82\ [245.24,340.40]$ | $36.76\ [30.39,43.13]$ |
| Active | $0.9500\ [0.9019,0.9981]$ | $0.0500\ [0.0019,0.0981]$ | $0.4544\ [0.3858,0.5229]$ | $269.06\ [226.22,311.89]$ | $32.15\ [26.79,37.52]$ |

这些结果只能证明 pipeline 和指标可运行，不能单独支持论文级 superiority claim。特别是 Active 与
GreedyPrior 的 SPL confidence intervals 大量重叠，而且 misleading-prior condition 中 Active
仍会被错误 prior 影响。

## 14. 论文级统计检验建议（尚未在代码中实现）

当前实现只有 descriptive statistics 和 normal CI。正式实验应保持 scenario/repetition 配对，
并增加：

1. 对 success/failure 的 paired McNemar test；
2. 对 SPL、time、distance、energy 的 paired bootstrap 95% CI；
3. 对非正态 paired differences 使用 Wilcoxon signed-rank test；
4. 多 policy、多 metric 比较使用 Holm correction；
5. 同时报 effect size，而不是只报告 $p$ value。

对任意两个 policy $A,B$，paired metric difference 为

$$
\delta_j=x_{A,j}-x_{B,j},
\qquad
\bar\delta=\frac{1}{m}\sum_{j=1}^{m}\delta_j.
$$

实验结论应基于 $\bar\delta$ 的 effect size 和 paired confidence interval，而不是只比较两组
independent CI 是否重叠。

## 15. LLM Prior 的单独评价

由于 LLM 输出目前只是 semantic relevance weights，必须单独验证其 calibration。对真实目标单元
$y_n$ 和预测 prior $b_0^{(n)}$，建议至少报告 negative log-likelihood：

$$
\mathrm{NLL}
=-\frac{1}{M}\sum_{n=1}^{M}\ln b_0^{(n)}(y_n),
$$

multi-class Brier score：

$$
\mathrm{Brier}
=\frac{1}{M}\sum_{n=1}^{M}\sum_{i=1}^{N}
\left(b_0^{(n)}(i)-\mathbb 1[y_n=i]\right)^2,
$$

以及 target-cell rank、top-$k$ recall 和 calibration error。只有当这些指标在 held-out tasks、
held-out maps 和不同 target categories 上得到验证后，才能声称 LLM prior 是 calibrated 或可泛化的。

## 16. 建议的核心 Ablation

为验证各数学项是否真正有用，至少比较：

$$
U_{\mathrm{full}},\quad
U-\lambda_{\mathrm{det}}q_v,\quad
U-\lambda_{\mathrm{IG}}\mathrm{IG}(v),\quad
U-\lambda_{\mathrm{nov}}N_t(v),\quad
U+\lambda_{\mathrm{travel}}C_d(v),
$$

以及 uniform prior、LLM prior、oracle prior、misleading prior。这样可以分别回答：

- LLM prior 是否提高目标发现效率；
- Bayesian negative evidence 是否能纠正错误 prior；
- expected information gain 是否提供额外收益；
- novelty 是否减少重复观察；
- travel penalty 是否减少时间、距离和能量；
- Active policy 是否优于只使用固定 prior 的 GreedyPrior。

## 17. 当前数学模型的边界

1. 单目标、单占据单元 categorical belief，不是多目标随机有限集模型。
2. V5 已覆盖 persistent distractor、episode-level correlated false alarm 和 Gaussian localization
   error，但 $P_D$/$P_{FA}$ 仍未按高度、距离、天气、遮挡和视角变化。
3. Candidate visibility 使用圆形 footprint，没有显式 ray casting 和 occlusion。
4. Active utility 是 one-step myopic policy，不是完整 POMDP value function。
5. LLM confidence 尚未经过 empirical calibration。
6. Smoke benchmark 的 normal CI 对 Bernoulli rate 和小样本并非最稳健，论文实验应使用
   Wilson/Clopper--Pearson 或 paired bootstrap。
7. 当前 benchmark 是 platform-neutral synthetic observation loop；Gazebo/PX4 实验必须单独报告
   collision、tracking error、control failure、realized energy 和 sensor latency。

因此，当前方法的准确表述应是：**一个由 LLM 生成 task-conditioned spatial prior、由 Bayesian
observation update 维护不确定性、并由可解释的 multi-objective active viewpoint utility 驱动的
闭环搜索框架。**
