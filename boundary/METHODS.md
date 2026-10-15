# Validation-designed policy libraries and reward-poisoning decisions

This experiment changes the candidate library while retaining the E2 estimator, confidence, attack, and deployment implementations. The unmodified implementation is [security/experiment.py](../security/experiment.py); its mathematical definitions and limitations are in [the E2 methods](../security/METHODS.md). E3 is a benchmark stress test, not a new estimator.

## Information separation

The official OptDigits training split is partitioned into 3,058 fitting and 765 validation examples using stratification and seed 417. Standardization, logistic-regression coefficients, and feature masks are fitted on the fitting split. Candidate mixture weights use validation rewards. The reward predictor is selected by validation log loss, as in E2.

`prepare.py freeze` reads only `optdigits.tra`. It saves the fitted parameters, validation predictions, mixture weights, protocol hash, construction-source hash, and timestamp. The separate `evaluate` command verifies those hashes before opening `optdigits.tes`. All six resulting candidates are retained. Evaluation values are not inputs to candidate construction or confidence allowances. E2 outcomes on this same dataset had already been inspected; this is a development benchmark, not an untouched external test of a final method.

Each policy is a probability distribution over the ten digit predictions. Its value is expected reward, not argmax classification accuracy. Reward is one for a correct sampled prediction and zero otherwise. The finite candidate list consists of continuously parameterized fitted models and their stochastic mixtures.

## Candidate construction

Keep the incumbent $\pi_0$ as full-feature logistic regression with $C=0.1$. Let $h$ be full-feature logistic regression with $C=10$. Three weaker parents $\ell_f$ are fitted using: (1) all features with $C=0.01$; (2) the top four image rows with $C=1$; and (3) the bottom four image rows with $C=1$. The 64 pixels are the original 8-by-8 digit representation.

On the validation set $\mathcal V$, calculate

$$
\widehat V_{\mathcal V}(\pi)
=\frac{1}{|\mathcal V|}\sum_{(x,y)\in\mathcal V}\pi(y\mid x).
$$

For each family $f$ and target $g\in\{-0.01,+0.01\}$, freeze

$$
t_{f,g}=\operatorname{clip}_{[0,1]}
\frac{\widehat V_{\mathcal V}(\pi_0)+g-\widehat V_{\mathcal V}(\ell_f)}
{\widehat V_{\mathcal V}(h)-\widehat V_{\mathcal V}(\ell_f)},
\qquad
\pi_{f,g}=(1-t_{f,g})\ell_f+t_{f,g}h.
$$

Clipping is reported rather than silently dropping a candidate. It was not needed in this run. This construction changes disagreement across contexts; it is not merely a smaller step along each original E2 candidate-minus-incumbent direction. Nevertheless, pairing the two targets within a family creates a further ordering effect, examined below.

## Logs, estimates, and decisions

For each of 100 seeds per setting, draw $n\in\{5000,20000\}$ independent contexts with replacement from the 1,797-example evaluation pool. Draw the action from

$$
\mu(a\mid x)=(1-\gamma)\pi_0(a\mid x)+\gamma/10,
\qquad \gamma\in\{0.2,0.5\}.
$$

A log row is $(x_i,a_i,r_i,\mu(a_i\mid x_i))$. The stored rewards are the only editable fields. The paired DR summand is

$$
Z_{ij}=\sum_a[\pi_j(a\mid x_i)-\pi_0(a\mid x_i)]\widehat q(x_i,a)
+c_{ij}[r_i-\widehat q(x_i,a_i)],
\qquad
c_{ij}=\frac{\pi_j(a_i\mid x_i)-\pi_0(a_i\mid x_i)}{\mu(a_i\mid x_i)}.
$$

The fitted DR rule uses the independently fitted reward predictor. Centered DR sets $\widehat q=1/2$; IPS sets it to zero. Known propensities make the unshrunk paired estimate unbiased under the specified logging experiment, regardless of reward-model error. Shrinkage changes the weights and adds the same worst-case bias allowance used in E2. This is our confidence adapter, not a reproduction of Su et al.'s entire selection algorithm.

Write a rule's nominal score as $S_j(r)=n^{-1}\sum_i c_{ij}r_i+b_j$, where $b_j$ includes the reward-model term and, where applicable, the confidence radius and shrinkage-bias allowance. The largest-drop correction for at most $k=\lfloor n\epsilon\rfloor$ reward edits is

$$
d_{ij}(r)=c_{ij}r_i-\min(c_{ij},0),\qquad
L_j(r)=S_j(r)-\frac1n\sum_{i\in\operatorname{Top}_k(d_j(r))}d_{ij}(r).
$$

The defender deploys the candidate with the highest score if that score exceeds the numerical threshold $10^{-9}$; otherwise it keeps $\pi_0$. A corrected confidence rule combines the statistical allowance with the edit correction. A corrected point-estimate rule has no sampling guarantee. Even on an unmodified log, a corrected rule uses its declared nonzero edit cap; it is not told whether an attack occurred.

The confidence allowances use label-free range and second-moment bounds over the known finite context distribution, with a union bound over the candidates. These are not claims about an unknown population beyond the evaluation pool. The primary library uses six candidates; each diagnostic subset uses three, so its confidence union factor changes accordingly.

## Threat model and attack verification

The attacker knows the log, frozen models, and full deployment rule and may replace at most $k$ rewards by arbitrary values in $[0,1]$. The oracle attacker also knows true finite-pool policy values; the defender does not. Contexts, actions, propensities, models, and the declared cap remain trusted. We test $\epsilon\in\{0,0.005,0.01\}$ under no edits, random flips, and an adaptive attack separately optimized for each rule.

The adaptive objective minimizes the true improvement of the policy selected by the complete gate, assigning zero improvement to fallback. Feasible search witnesses use binary flips. Deterministic score envelopes and checked LP dual certificates cover all real-valued reward replacements within the cap. A feasible attack gives an upper bound on minimum attainable utility; the certificate gives a lower bound. Every unresolved gap is retained. A harmful witness demonstrates a failure; absence of one is not a certificate unless all harmful candidates are excluded.

## Why removing a sibling is a useful diagnostic

Within a family,

$$
\pi_{f,+}-\pi_{f,-}=(t_{f,+}-t_{f,-})(h-\ell_f).
$$

Consequently, a nominal affine gate may preserve their ordering even when both candidates are close in average value to the incumbent. Let $a_i=c_{i,+}-c_{i,-}$ and $m(r)=S_+(r)-S_-(r)$. The exact worst pairwise margin is

$$
\min_{\substack{r'\in[0,1]^n\\\|r'-r\|_0\le k}}m(r')
=m(r)-\frac1n\sum_{i\in\operatorname{Top}_k(a_i r_i-\min(a_i,0))}
[a_i r_i-\min(a_i,0)].
$$

A strictly positive margin excludes the worse sibling from selection while the better sibling is available. `diagnostics.py` calculates this quantity for every log, budget, method, and family. This diagnosis is separate from general claims about arbitrary policy libraries.

E3b evaluates all six ways to choose one candidate from each of the three families while including both validation target signs. The codes `m` and `p` denote the negative and positive validation targets, in regularization/top/bottom order. The all-positive and all-negative combinations are excluded by the stated question, not by their test outcomes. The diagnostic was specified after inspecting E3 timing results; it is explicitly exploratory. Every mixed library and every grid cell is reported. Policies, attack budgets, and logs are unchanged. Point-estimate comparisons isolate candidate availability without a confidence-union-factor change.

## Metrics and interpretation

The primary axes are harmful-deployment probability $P(\Delta_{\widehat j}<0)$ and mean true improvement $E[\Delta_{\widehat j}]$, with $\Delta_j=V_T(\pi_j)-V_T(\pi_0)$ and fallback improvement zero. Report fallback, clean improvement sacrificed, certified utility floors, and attack gaps alongside them. Zero-edit strata must be separated from positive-budget strata: positive average improvement can otherwise come entirely from the no-attack setting.

Uncertainty calculations pair methods on the same logs and bootstrap 100 seed clusters. The six diagnostic libraries reuse the same 400 logs; they are not additional independent datasets or safety trials. Pointwise cell intervals are descriptive, not a multiple-comparison claim of optimality. A finite empirical frontier, even with exact attack solutions on stored logs, does not prove global Pareto optimality or a new contribution relative to the literature.

## Provenance and ethics

The starting point is the [released OptDigits policy-selection benchmark of Kuzborskij et al. (2021)](https://proceedings.mlr.press/v130/kuzborskij21a.html). [Dudik et al. (2011)](https://arxiv.org/abs/1103.4601) supply DR evaluation; [Su et al. (2020)](https://proceedings.mlr.press/v119/su20a.html) supply the optimistic shrinkage map; [Lobo et al. (2022)](https://proceedings.mlr.press/v180/lobo22a.html) study data poisoning of off-policy evaluation. The project-specific work here is validation-based candidate construction, incumbent-relative deployment evaluation, the matched reward-only attack/correction implementation, and the audited library ablation. We do not claim to reproduce Lobo's influence-function attacks or every baseline from these papers. The E2 ESLB supplement is preserved but was not rerun for E3.

Data: [UCI OptDigits, E. Alpaydin and C. Kaynak](https://doi.org/10.24432/C50P49), CC BY 4.0. All attacks operate on local copies of this public benchmark. No external deployed system is attacked.
