# Related Work, References, and Research Gaps for GSI Search Intelligence

> Purpose: establish a defensible reference map for developing this repository into a research paper. Audited on August 4, 2026.
>
> The repository does not yet contain a formal bibliography. This document distinguishes methodological foundations, closely related work, engineering projects, and patent prior art. It does not claim that GSI reproduces methods that it does not implement.

## 1. Research Positioning

The current Search Intelligence method can be summarized as a closed-loop active-search framework that:

1. obtains a task-conditioned semantic spatial prior from an LLM;
2. represents uncertain target location as a discrete Bayesian belief;
3. scores candidate viewpoints using detection probability, expected information gain, observation novelty, and travel cost;
4. updates the posterior after positive or negative observations; and
5. stops after verified detection or resource-budget exhaustion.

The work intersects ObjectGoal Navigation, semantic exploration, Bayesian target search, active visual search, Next-Best-View planning, Coverage Path Planning, semantic mapping, open-vocabulary detection, POMDP decision-making, and embodied-navigation evaluation.

## 2. Core References

### R1. PONI: Potential Functions for ObjectGoal Navigation With Interaction-Free Learning

- **Citation**: Santhosh Kumar Ramakrishnan, Devendra Singh Chaplot, Ziad Al-Halah, Jitendra Malik, and Kristen Grauman. CVPR 2022, pp. 18890–18900. arXiv:2201.10029.
- **Paper**: https://openaccess.thecvf.com/content/CVPR2022/html/Ramakrishnan_PONI_Potential_Functions_for_ObjectGoal_Navigation_With_Interaction-Free_Learning_CVPR_2022_paper.html
- **Code**: https://github.com/srama2512/PONI
- **Main contribution**: PONI separates where to search from how to navigate. It predicts object and area potential functions from a top-down semantic map, selects frontier-based long-term goals, and trains the potential network from passive map data without interactive RL.
- **Relationship to GSI**: both methods use semantic maps to answer where to look and separate high-level spatial decisions from low-level navigation.
- **Difference**: GSI does not implement the PONI potential-function network, frontier supervision, or ObjectNav local policy. GSI currently uses LLM-generated label weights, a Bayesian target belief, and an explicit multi-term viewpoint utility.
- **How to cite it**: PONI should be a central related-work reference, but GSI should not be described as a PONI implementation.

### R2. Object Goal Navigation using Goal-Oriented Semantic Exploration

- **Citation**: Devendra Singh Chaplot, Dhiraj Prakashchand Gandhi, Abhinav Gupta, and Ruslan Salakhutdinov. NeurIPS 2020, Vol. 33. arXiv:2007.00643.
- **Paper**: https://papers.nips.cc/paper_files/paper/2020/hash/2c75cf2681788adaca63aa95ae028b22-Abstract.html
- **Code**: https://github.com/devendrachaplot/Object-Goal-Navigation
- **Main contribution**: SemExp builds an episodic semantic map, learns object-layout priors with a goal-conditioned long-term policy, and uses a deterministic local policy to reach the selected goal.
- **Relationship to GSI**: it is closely related to SemanticGridBuilder, task-conditioned priors, and the separation of semantic decision-making from navigation.
- **Difference**: SemExp learns its prior through an RL policy. GSI obtains semantic relevance weights from an LLM and maintains an explicit Bayesian target-location belief.

### R3. Probabilistic Robotics

- **Citation**: Sebastian Thrun, Wolfram Burgard, and Dieter Fox. MIT Press, 2005. ISBN 9780262201629.
- **Link**: https://mitpress.mit.edu/9780262201629/probabilistic-robotics/
- **Main contribution**: a unified treatment of Bayes filters, sensor and motion models, belief updates, localization, and mapping under uncertainty.
- **Relationship to GSI**: this is the direct foundation for BeliefMap, BinarySensorModel, BayesianBeliefUpdater, and the explicit treatment of uncertain perception.

### R4. Optimal Eye Movement Strategies in Visual Search

- **Citation**: Jiri Najemnik and Wilson S. Geisler. Nature 434, 387–391, 2005. DOI: 10.1038/nature03390.
- **Paper**: https://www.nature.com/articles/nature03390
- **Main contribution**: an ideal Bayesian visual searcher uses scene statistics, visibility, and the current posterior to select the next fixation that is most informative about target location.
- **Relationship to GSI**: it directly matches the loop of unknown target location, viewpoint-dependent sensing, Bayesian update, and next-observation selection.
- **Difference**: the paper studies human eye movements and two-dimensional visual search, while GSI controls spatial viewpoints for a robot or UAV.

### R5. Information-Theoretic Based Target Search with Multiple Agents

- **Citation**: Minkyu Kim, Ryan Gupta, and Luis Sentis. arXiv:2107.12715, 2021.
- **Paper**: https://arxiv.org/abs/2107.12715
- **Main contribution**: candidate paths are generated from global waypoints and local frontiers, ranked by information gain, and executed by heterogeneous robot teams in simulation and real environments.
- **Relationship to GSI**: this is one of the closest robotics references for candidate selection through expected information gain and is relevant to future multi-UAV work.
- **Difference**: the current SearchSession does not implement the paper's sequential multi-robot coordination.

### R6. A Mathematical Theory of Communication

- **Citation**: Claude E. Shannon. Bell System Technical Journal 27, 1948, pp. 379–423 and 623–656.
- **Paper**: https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
- **Main contribution**: the mathematical foundations of entropy, conditional entropy, and mutual information.
- **Relationship to GSI**: directly supports entropy_nats, binary entropy, information_gain_nats, and the mutual-information term in the active-search utility.

### R7. On Information and Sufficiency

- **Citation**: Solomon Kullback and Richard A. Leibler. Annals of Mathematical Statistics 22(1), 79–86, 1951. DOI: 10.1214/aoms/1177729694.
- **Paper**: https://doi.org/10.1214/aoms/1177729694
- **Main contribution**: introduces information divergence between probability distributions.
- **Relationship to GSI**: directly supports the KL-divergence diagnostic used to quantify belief change after an observation.

### R8. Coverage of Known Spaces: The Boustrophedon Cellular Decomposition

- **Citation**: Howie Choset. Autonomous Robots 9, 247–253, 2000. DOI: 10.1023/A:1008958800904.
- **Paper**: https://doi.org/10.1023/A%3A1008958800904
- **Main contribution**: decomposes known free space into cells that can each be covered by back-and-forth motions.
- **Relationship to GSI**: supports the lawnmower and back-and-forth CoveragePolicy baseline.
- **Boundary**: GSI does not reproduce the complete boustrophedon cellular-decomposition algorithm; it implements a practical geometric coverage route inspired by classical CPP.

### R9. Coverage for Robotics – A Survey of Recent Results

- **Citation**: Howie Choset. Annals of Mathematics and Artificial Intelligence 31, 113–126, 2001. DOI: 10.1023/A:1016639210559.
- **Paper**: https://doi.org/10.1023/A%3A1016639210559
- **Main contribution**: surveys heuristic, randomized, and cellular-decomposition approaches to coverage planning.
- **Relationship to GSI**: positions the coverage baseline and distinguishes complete geometric coverage from target-conditioned active search.

### R10. On Evaluation of Embodied Navigation Agents

- **Citation**: Peter Anderson et al. arXiv:1807.06757, 2018.
- **Paper**: https://arxiv.org/abs/1807.06757
- **Main contribution**: standardizes embodied-navigation tasks and metrics and introduces Success weighted by Path Length.
- **Relationship to GSI**: directly supports the SPL metric in the search benchmark.
- **Boundary**: GSI defines the shortest distance as the distance from the initial pose to the nearest candidate capable of observing the target. The paper must explicitly call this a search-adapted SPL.

## 3. Closely Related and Extending Work

### R11. Active Object Tracking using Context Estimation

- **Citation**: Minkyu Kim and Luis Sentis. arXiv:1912.06754, 2019.
- **Paper**: https://arxiv.org/abs/1912.06754
- **Main contribution**: combines contextual information, a Dynamic Bayesian Network, an information-theoretic utility, and a high-level POMDP for finding missing or occluded targets.
- **Relationship to GSI**: both combine contextual priors, posterior inference, and information gain. It also highlights GSI's current boundary: a single static target with a discrete cell belief.

### R12. Coordinated Search for a Lost Target in a Bayesian World

- **Citation**: Timothy Chung and Joel Burdick. Advanced Robotics 18(10), 2004.
- **Paper**: https://www.tandfonline.com/doi/abs/10.1163/1568553042674707
- **Main contribution**: multiple autonomous sensor platforms maintain a Bayesian target density and coordinate a decentralized search.
- **Relationship to GSI**: a classical reference for extending a single search session to shared multi-robot beliefs and coordinated viewpoints.

### R13. Planning and Acting in Partially Observable Stochastic Domains

- **Citation**: Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Artificial Intelligence 101(1–2), 99–134, 1998. DOI: 10.1016/S0004-3702(98)00023-X.
- **Paper**: https://doi.org/10.1016/S0004-3702(98)00023-X
- **Main contribution**: foundational belief-state planning for POMDPs.
- **Relationship to GSI**: target search naturally admits a POMDP formulation, but ActiveSearchPolicy is currently a one-step myopic utility rather than a full POMDP value-function solver.

### R14. FUEL: Fast UAV Exploration Using Incremental Frontier Structure and Hierarchical Planning

- **Citation**: Boyu Zhou, Yichen Zhang, Xinyi Chen, and Shaojie Shen. IEEE RA-L 6(2), 779–786, 2021. DOI: 10.1109/LRA.2021.3051563.
- **Paper**: https://doi.org/10.1109/LRA.2021.3051563
- **Code**: https://github.com/HKUST-Aerial-Robotics/FUEL
- **Main contribution**: hierarchical UAV exploration with an incremental frontier structure, global coverage, local viewpoint refinement, and minimum-time trajectories.
- **Relationship to GSI**: relevant to UAV viewpoint generation, the separation of search decisions from trajectory execution, and real-flight validation.
- **Difference**: GSI searches for a target inside a known search geometry; it is not a frontier-based 3D mapping system and does not optimize minimum-time trajectories.

### R15. Frontier-Based Exploration Using Multiple Robots

- **Citation**: Brian Yamauchi. Autonomous Agents 1998, pp. 47–53. DOI: 10.1145/280765.280773.
- **Paper**: https://doi.org/10.1145/280765.280773
- **Main contribution**: treats boundaries between known free space and unknown space as exploration goals and extends frontier exploration to multiple robots.
- **Relationship to GSI**: a classical comparison for candidate-goal generation. GSI currently generates candidates from known SearchGrid cells instead of map frontiers.

## 4. Open-Vocabulary Perception

### R16. YOLO-World: Real-Time Open-Vocabulary Object Detection

- **Citation**: Tianheng Cheng et al. CVPR 2024, pp. 16901–16911.
- **Paper**: https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html
- **Code**: https://github.com/AILab-CVC/YOLO-World
- **Main contribution**: extends YOLO to efficient open-vocabulary and zero-shot detection through vision-language pretraining.
- **Relationship to GSI**: the ROS bridge documentation names YOLO-World as a future detector, and SearchTarget query/prompts provide a compatible task interface.
- **Boundary**: it is not integrated into the current core experiments, so current results must not be described as open-vocabulary search.

### R17. Grounding DINO

- **Full title**: Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.
- **Citation**: Shilong Liu et al. ECCV 2024. arXiv:2303.05499.
- **Paper**: https://arxiv.org/abs/2303.05499
- **Code**: https://github.com/IDEA-Research/GroundingDINO
- **Main contribution**: combines language grounding with a transformer detector to localize open-set targets specified by free-form text.
- **Relationship to GSI**: it is another planned detector backend and directly matches the natural-language target-query interface.

### R18. Semantic Mapping in SemExp and PONI

- **Main idea**: project first-person semantic predictions using RGB-D and pose into an allocentric top-down semantic map.
- **Relationship to GSI**: related to projecting visible ground points or point clouds into SearchGrid cells.
- **Difference**: SemanticGridBuilder currently annotates the grid mainly from a known scene graph rather than constructing a complete map online from RGB-D. This privileged-map assumption must be disclosed.

## 5. Method-to-Reference Mapping

| Repository idea | Recommended reference | Relationship | Required boundary |
| --- | --- | --- | --- |
| Categorical target-location belief | Thrun et al., 2005 | Direct foundation | One target, one cell |
| Binary positive/negative likelihood | Thrun et al., 2005 | Direct foundation | Simplified detection and false-alarm model |
| Shannon entropy | Shannon, 1948 | Direct | Natural logarithm, measured in nats |
| KL divergence | Kullback and Leibler, 1951 | Direct | Posterior relative to prior |
| Mutual-information viewpoint gain | Shannon, 1948; Najemnik, 2005; Kim, 2021 | Direct and closely applied | Binary visible/not-visible mutual information |
| Myopic NBV | Active visual search and NBV literature | Closely related | Not non-myopic trajectory planning |
| Task-conditioned semantic prior | SemExp, 2020; PONI, 2022 | Closely related | GSI uses LLM label weights |
| Confidence mixing with uniform | Robust-prior engineering | GSI-specific design | Requires ablation and calibration |
| Novelty mass | Exploration bonus and coverage gain | Conceptual relation | The exact formula is a GSI design |
| Travel penalty | Informative path planning | Conceptual relation | Current cost is straight-line distance |
| Lawnmower baseline | Choset, 2000/2001 | Direct method family | Not full exact decomposition |
| Multi-observation verification | Sequential detection logic | Engineering relation | Threshold rules, not SPRT, JPDA, or RFS |
| SPL | Anderson et al., 2018 | Direct adaptation | Search-specific shortest-distance definition |
| POMDP | Kaelbling et al., 1998 | Theoretical superclass | Current implementation is not a POMDP solver |

## 6. Open-Source Projects for Baselines and Reproduction

| Project | Repository | Use in a paper |
| --- | --- | --- |
| PONI | https://github.com/srama2512/PONI | Main semantic ObjectNav comparison |
| SemExp | https://github.com/devendrachaplot/Object-Goal-Navigation | Semantic-map and goal-conditioned exploration comparison |
| FUEL | https://github.com/HKUST-Aerial-Robotics/FUEL | 3D UAV frontier, NBV, and trajectory comparison |
| Habitat-Lab | https://github.com/facebookresearch/habitat-lab | Standard ObjectNav environments and evaluation |
| YOLO-World | https://github.com/AILab-CVC/YOLO-World | Open-vocabulary perception backend |
| Grounding DINO | https://github.com/IDEA-Research/GroundingDINO | Text-conditioned detection backend |
| PX4-Autopilot | https://github.com/PX4/PX4-Autopilot | UAV SITL and flight control |
| MAVROS | https://github.com/mavlink/mavros | ROS-to-MAVLink integration |
| Gazebo | https://github.com/gazebosim/gz-sim | Physics and sensor simulation |

Formal papers should prefer each project's official paper, Zenodo DOI, or software citation over a bare GitHub URL.

## 7. Relevant Patent Prior Art

> The following items are included only for prior-art awareness. They do not imply that GSI uses or infringes any claim, and they are not a freedom-to-operate or legal analysis.

### P1. CN113505646A/B — Target Searching Method Based on a Semantic Map

- **Applicant**: Tsinghua University. Priority date: June 10, 2021. Granted version: CN113505646B.
- **Patent**: https://patents.google.com/patent/CN113505646A/en
- **Main content**: builds semantic relations between a target and parent objects, reconstructs a semantic map from multi-view RGB-D data, visits navigation points near related parent objects, and updates the semantic relation graph.
- **Relationship to GSI**: semantic mapping, object co-occurrence, and search-point selection.
- **Difference**: GSI uses LLM label weights and Bayesian NBV rather than the patent's relation-graph search procedure.

### P2. CN119223305A — Multi-Robot Visual Semantic Navigation Method

- **Patent**: https://patents.google.com/patent/CN119223305A/en
- **Main content**: robots communicate to improve semantic maps, and planning combines target-existence probability with uncertainty when selecting the next target location.
- **Relationship to GSI**: closely related to beliefs, exploration-exploitation trade-offs, and online replanning; it also covers shared multi-robot maps, which SearchSession does not implement.

### P3. US9696430B2 — Locating a Target Using an Autonomous UAV

- **Patent**: https://patents.google.com/patent/US9696430B2/en
- **Main content**: an autonomous UAV searches an area of interest for a target whose location is initially unknown.
- **Relationship to GSI**: similar application setting, but not a direct source for Bayesian semantic active search.

### P4. CN120162394B — Semantic Mapping with a Multimodal Large Model and Graph RAG

- **Patent**: https://patents.google.com/patent/CN120162394B/en
- **Main content**: combines a multimodal model, semantic database, object co-occurrence graph, and Graph RAG for semantic-map construction.
- **Relationship to GSI**: related to turning task and environmental semantics into spatial knowledge.
- **Difference**: GSI currently has no Graph RAG or online multimodal map-construction pipeline.

## 8. GSI-Specific Design Elements

The following combinations should be presented as GSI design choices and experimentally justified instead of being attributed to one prior paper:

1. the platform-neutral SearchTask, SearchState, SearchObservation, and SearchOutcome contracts;
2. projection of LLM semantic-label weights onto a grid followed by confidence mixing with a uniform distribution;
3. a utility combining detection probability, binary mutual information, quality-weighted novelty, and normalized travel cost;
4. unified localized-positive, unlocalized-positive, and negative evidence;
5. modulation of effective detection probability by detection confidence and observation quality;
6. verification mode based on confirmation count, persistence, and localization error;
7. strict separation of simulator ground truth behind the observation interface; and
8. reuse of SearchSession in synthetic benchmarks, semantic simulation, and ROS/Gazebo execution.

Combining established modules is not sufficient by itself to establish novelty. The paper needs a precise research question and experiments showing benefits that cannot be explained by any single pre-existing component.

## 9. Recommended Related-Work Structure

### 9.1 Semantic Object-Goal Navigation

Discuss SemExp and PONI, including semantic maps, spatial co-occurrence priors, long-term goals, and local navigation. Then distinguish outdoor UAV target search, LLM priors, and explicit Bayesian sensing from learned indoor ObjectNav policies.

### 9.2 Bayesian and Information-Theoretic Active Search

Introduce Bayes filtering, Bayesian visual search, and information-theoretic target search. Position GSI's combination of an LLM prior, negative evidence, sensor quality, and confirmation-aware verification.

### 9.3 Coverage, Frontier, and UAV Exploration

Discuss coverage planning, frontier exploration, and FUEL. Treat coverage as a baseline and clarify that GSI optimizes target discovery rather than unknown-space mapping.

### 9.4 Open-Vocabulary Perception

Discuss YOLO-World and Grounding DINO. If experiments still use a color detector, open-vocabulary perception can only be described as an intended backend, not a demonstrated contribution.

## 10. Work Required for a Publication-Quality Paper

### A. Fix the Research Question and Contributions

1. Focus the paper on LLM task-conditioned priors plus Bayesian active UAV search rather than the entire GSI stack.
2. State two to four falsifiable contributions, each supported by a method section and experiments.
3. Establish structural differences from PONI, SemExp, and conventional information-theoretic search.

### B. Complete the Method

1. Formally define state, action, observation, likelihood, budget, termination, and objective.
2. Provide complete algorithm pseudocode.
3. Explain the scale, weights, and tuning procedure for every utility term.
4. Calibrate or experimentally justify detection probability, false-alarm probability, observation quality, and detector confidence.
5. Add camera frustums, ray casting, and occlusion-aware visibility.
6. Replace straight-line travel cost with feasible path length or flight time.

### C. Add Strong Baselines

1. Coverage and Random, which already exist.
2. Greedy current-posterior mass.
3. Information-gain-only NBV.
4. A frontier baseline for unknown-map experiments.
5. A PONI- or SemExp-style semantic baseline, or evaluation on a directly comparable task.
6. Oracle and uniform priors.
7. YOLO-World or Grounding DINO if open-vocabulary search is claimed.

### D. Run Required Ablations

1. Remove the LLM prior.
2. Remove Bayesian negative evidence.
3. Remove detection, information gain, novelty, and travel terms one at a time.
4. Remove verification mode.
5. Compare confidence mixing with direct normalization of LLM weights.
6. Isolate localized-positive, unlocalized-positive, and negative evidence.
7. Test sensitivity to grid resolution, candidate stride, altitude, field of view, and budget.

### E. Expand Data and Experiments

1. Use multiple maps, target categories, target placements, and prior conditions.
2. Hold out maps, categories, and language instructions.
3. Report synthetic, Gazebo/PX4 SITL, and real-robot results separately.
4. Test real operation across lighting, altitude, occlusion, and sensor latency.
5. Categorize false alarms, misses, bad priors, localization failures, control failures, and budget exhaustion.

### F. Evaluate the LLM Prior Separately

1. Target-cell negative log-likelihood and multiclass Brier score.
2. Target-cell rank, top-k recall, calibration curves, and expected calibration error.
3. Stability across LLMs, prompts, temperatures, and semantic-label inventories.
4. Token use, latency, and monetary cost.
5. Recovery speed under misleading or adversarial priors.

### G. Improve Metrics and Statistics

1. Report success, false-positive, false-negative, and search-adapted SPL.
2. Report time, distance, and energy to detection.
3. Report posterior NLL, Brier score, entropy, and localization error.
4. Use paired bootstrap confidence intervals.
5. Use paired McNemar tests for success.
6. Use paired Wilcoxon tests for continuous metrics, or paired t-tests after checking assumptions.
7. Apply Holm correction to multiple comparisons and report effect sizes.
8. Use Wilson, Clopper–Pearson, or paired-bootstrap intervals for small Bernoulli samples instead of only normal intervals.

### H. Ensure Reproducibility

1. Freeze commit hashes, container images, dependency versions, and seeds.
2. Release configurations, scenario generators, raw traces, and aggregation scripts.
3. Provide one-command reproduction for main and ablation tables.
4. Separate training, tuning, and test maps.
5. Report compute, runtime, LLM APIs, and retry rules.
6. Add a license and provenance table for data, models, maps, and third-party code.

## 11. Recommended Figures and Tables

1. **Figure 1**: task to semantic prior, belief, viewpoint, observation, and posterior.
2. **Figure 2**: prior/posterior heatmaps and viewpoint evolution for one episode.
3. **Figure 3**: behavior under correct, uniform, and misleading priors.
4. **Table 1**: Coverage, Random, Greedy, IG-only, semantic baseline, and full method.
5. **Table 2**: prior, utility, update, and verification ablations.
6. **Table 3**: generalization across maps, targets, and sensor stress.
7. **Table 4**: synthetic, Gazebo/PX4, and real-world sim-to-real gaps.
8. **Appendix**: derivations, hyperparameters, prompts, scenarios, statistics, and failure cases.

## 12. Recommended Paper Narrative

The paper should not claim to be the first use of Bayesian search or information-gain NBV. A more defensible narrative is:

> Existing semantic ObjectNav methods often learn long-term goals or potentials in indoor scenes, while classical Bayesian target search usually assumes that the prior and sensor model are given. We study how natural-language tasks and a public semantic map can be converted into an auditable spatial prior, then corrected online from real observations during open-outdoor UAV search while jointly accounting for information gain, motion cost, sensor quality, and repeated confirmation.

This narrative still requires strong baselines, complete ablations, held-out multi-map evaluation, rigorous statistics, and real-system evidence.

## 13. Priorities

### Required Before Submission

1. Freeze the claims and contributions.
2. Complete the core related work.
3. Run all core ablations.
4. Add strong semantic and NBV baselines.
5. Build a held-out multi-map benchmark.
6. Evaluate LLM-prior calibration separately.
7. Recompute results with paired statistics.
8. Run sufficient Gazebo/PX4 trials and preferably real-robot experiments.

### Follow-Up Extensions

1. A real open-vocabulary detector.
2. Occlusion-aware three-dimensional viewpoint generation.
3. Non-myopic POMDP or informative path planning.
4. Dynamic targets and multi-target beliefs.
5. Multi-UAV belief sharing and coordinated allocation.
6. Learned utility weights or an end-to-end policy.
