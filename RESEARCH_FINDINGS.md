# Reward poisoning: findings and the next research claim

The experiments now support a focused question: **when can a small protected sample extract more useful decisions from an existing, potentially poisoned log than spending the same resources on protected data alone?** The strongest evidence is in a controlled setting with a fixed historical log. Unrestricted acquisition creates a different problem, where simply collecting more ordinary data can win. Both results are necessary to define a defensible contribution.

## What the experimental sequence has established

| Stage | Finding | Research consequence |
|---|---|---|
| [E0: released OptDigits baseline](README.md#e0-a-clean-baseline-for-off-policy-policy-selection) | The released confidence-based selector chose the ideal reference in all ten trials. Differences from the paper remain documented. | Establishes an auditable clean baseline; it is not evidence of poisoning robustness or a faithful reproduction of every paper comparison. |
| [E2: independently trained policies](security/README.md) | The complete 400-log experiment found no harmful selections even without protection. | A test with no exposed vulnerability cannot demonstrate a safety benefit. Learned policies were too separated for that question under the tested attack. |
| [E3: policies near the incumbent](boundary/README.md) | The benchmark exposed vulnerability, but the corrected gate retained the incumbent at every tested nonzero budget. | Conservative safety can remove all useful improvement. Always retaining the incumbent is an essential baseline. |
| [Targeted OptDigits acquisition](design/README.md) | At 0.5% corruption, the reported gain was 0.910 pp under feasible attacks, with a certified utility floor of 0.863 pp. | Changing which evidence is collected can preserve useful improvement. This is prospective collection with explicit acquisition assumptions, not a repair of historical data. |
| [E6–E7: confidence allocation](controlled/README.md) | Influence-based allocation helped when the better policy had low influence but failed after reversing that relationship. | Policy sensitivity alone is insufficient for allocating testing error. A method must not implicitly assume that low influence identifies a good policy. |
| [E8: source weighting](evidence/README.md) | Weighting independent trusted evidence improved utility over pooling in both geometries and survived an oracle risk-ceiling diagnostic. Cheap trusted-only acquisition won; allowing reference poisoning removed the weighting advantage. | Source weighting is a plausible mechanism. Reference integrity, acquisition costs, and calibration must remain explicit. Independence does not imply integrity. |
| [E9: acquisition choice](acquisition_choice/README.md) | The rule can buy only trusted or only ordinary data. Weighting is useful with a fixed historical log, but stronger acquisition alternatives and a failure of the selection objective limit the claim. | The next contribution should target useful decisions under a declared resource model. Minimizing an approval barrier does not establish optimal utility. |

The OptDigits experiments use real digit features with constructed bandit feedback. E6–E9 use a synthetic population with known policy values and complete attacks. These are complementary forms of evidence and should remain distinct. The sequence is a reproducible mechanism study; a general learned-policy result and a verified novelty claim remain to be established.

## The result worth presenting now

In E9, start with a common 19,000-row historical log. A white-box attacker can edit 200 ordinary rewards; protected rewards cannot be edited. An additional budget of 10,000 buys 2,000 trusted rows at price ratio 5. At the common analytic harm ceiling of 1%, weighted evidence gives **0.257 pp** of worst-attack improvement compared with **0.156 pp** for trusted-only evaluation and **0.219 pp** for pooling. When the good policy has high rather than low influence, the gains are **2.319**, **1.411**, and **1.979 pp**, respectively. The reversal is important: the result is not confined to the geometry that favors the older influence heuristic.

The prespecified target of retaining 75% of the best candidate's improvement is met by weighting at additional budget **10,000** in both geometries. Trusted-only evaluation first meets it at **20,000** on the tested grid. Simultaneous intervals support success for weighting and failure for trusted-only at 10,000. This is evidence for a cost–utility benefit under a common safety guarantee. It does not determine exact minimum acquisition costs or establish equality of actual harmful-deployment risk.

The unresolved comparison is with influence allocation in the first geometry: the improvement is **0.018 pp**, but the conservative simultaneous difference lower bound is **−0.008 pp**. The new joint acquisition rule also makes exactly the same choices as fixed-design weighting in this setting. The representative result should therefore be presented as a source-weighting result, with the automatic chooser as an exploratory extension.

Two counterexamples must accompany the positive result. First, with all observations purchased at total budget 69,000 and price ratio 50, ordinary-only acquisition obtains **0.300 pp**, exceeding the fixed mixture's **0.212 pp** under the 200-edit threat. Second, at total budget 10,000 and price ratio 20, the chooser obtains a slightly negative gain while trusted-only acquisition improves by **0.030 pp**, or **0.274 pp** in the reversal. Cost accounting and the choice of optimization objective materially change the conclusion.

## Connection to existing work

[Kuzborskij et al. (2021)](https://proceedings.mlr.press/v130/kuzborskij21a.html) develop confidence-based off-policy selection using self-normalized importance weighting and an Efron–Stein bound. E0 runs their released implementation. The relevant inherited idea is to select policies using uncertainty-aware scores. E9's exact binomial gate is a separate controlled construction, not their algorithm extended by a proven poisoning correction.

[Lobo et al. (2022)](https://proceedings.mlr.press/v180/lobo22a.html) construct influence-function-based poisoning attacks that increase off-policy value-estimation error. Our question concerns the final deployment decision: does poisoning induce a policy worse than the incumbent, and how much useful improvement can a defense retain? E9's exact endpoint attacks address its restricted gate; they do not reproduce Lobo et al.'s general attack framework.

[Charikar, Steinhardt, and Valiant (2017)](https://arxiv.org/abs/1611.02315) study semi-verified learning with a small trusted sample alongside untrusted data. Trusted evidence is therefore an established research ingredient. Our possible extension is a decision-specific treatment of its acquisition cost, adaptive reward poisoning, and safe policy improvement. A paper must establish what is new in the resulting guarantee or acquisition rule; combining familiar ingredients alone is insufficient.

The direct chain is thus: confidence-based policy selection supplies the decision framework; poisoning work supplies the adversarial concern; semi-verified learning motivates protected evidence; our experiments isolate when that evidence is useful and when its cost or trust assumptions destroy the advantage. General OPE, robust estimation, and evidence-acquisition methods must be compared in a common learned-policy setting before claiming broad superiority.

## An explicit Pareto claim and how to test it

Here “Pareto” means a frontier among competing objectives, not the 80/20 rule. Let $d$ denote a complete acquisition-and-deployment procedure, $D$ its clean random data, and $\mathcal A(D)$ the permitted reward modifications after the attacker observes those data. Let $D^e$ be the data after modification $e$, and let $\Delta(\pi)=V(\pi)-V(\pi_0)$ be a policy's population gain over incumbent $\pi_0$, with $\Delta(\pi_0)=0$. Define

$$
H(d)=\Pr_D\{\exists e\in\mathcal A(D):\Delta(d(D^e))<0\},
\qquad
U(d)=\mathbb E_D\left[\min_{e\in\mathcal A(D)}\Delta(d(D^e))\right].
$$

Let $C(d)$ be acquisition cost under one specified resource model. For outcome-adaptive acquisition, the protocol must declare whether $C$ means expected cost or a hard upper budget. A rule $d_1$ dominates $d_2$ if it has no greater $H$ or $C$ and no smaller $U$, with at least one strict improvement.

The practical target is an **extension of the tested frontier**. Fix a risk limit $h_0$, cost limit $c_0$, and comparator family $\mathcal B$ before confirmation. Show that our frozen rule satisfies these limits and exceeds the best feasible comparator:

$$
H(d_\star)\le h_0,\qquad C(d_\star)\le c_0,\qquad
U(d_\star)>\sup_{b\in\mathcal B:\,H(b)\le h_0,\,C(b)\le c_0}U(b).
$$

This claim has a concrete meaning: within a declared comparison family and regime, no tested alternative can reach our utility without accepting more risk or cost. It does not assert optimality over all algorithms. A single frontier extension does not prove that our entire method family is Pareto-optimal; nondominance and superiority over one baseline are also different claims.

There are two legitimate safety axes. A **guarantee–utility–cost frontier** compares a specified valid risk bound, such as the uniform 1% ceiling, against utility and cost. A **risk–utility–cost frontier** compares actual worst-attack harm probabilities. E9 already provides evidence for the former against several comparators. The latter needs comparable risk estimates or exact evaluation, and the remaining influence comparison needs stronger evidence. Equal nominal significance levels alone do not settle that comparison.

### A decisive next experiment

1. **Freeze the scientific question.** Use the historical-log model as the primary resource setting, and total-purchase accounting as a separate robustness study. Prespecify prices, both policy geometries, fixed and rate-based corruption, useful-gain targets, and a bounded set of cost and risk levels. Keep the current outcomes for development; confirmation must use fresh data.
2. **Give every comparator its best fair configuration.** Include always retain, ordinary-only, trusted-only, pooling, influence allocation, independent-pilot rules, fixed-design weighting, and the chooser. Sweep each comparator's tuning parameters and acquisition plans under the same information and cost constraints. Include random mixtures of deployable baseline rules in the comparison family, charging the declared expected or worst-case cost consistently; otherwise a claimed frontier can miss a simple randomized alternative.
3. **Improve the objective before adding complexity.** The current rule minimizes a mean threshold. A stronger rule should optimize a justified lower bound on worst-attack utility, or the probability of a useful deployment at a declared target effect size. Trusted-only and ordinary-only acquisition remain admissible choices. If a pilot estimates effect sizes, charge its observations and use independent confirmation, conditional validity, or a proved simultaneous bound. Never use confirmation labels or known policy truth to select acquisition or weights.
4. **Match the safety constraint without an oracle defender.** Learn operating points on separate development data or use valid uniform calibration. Freeze them before confirmation. The evaluator may use exact synthetic truth to assess risk, but the deployed procedure must not use the true bad-policy probability. Report both valid guarantees and empirical or exact worst-attack harm. For the practical frontier, resolve comparators whose uncertainty intervals straddle the risk ceiling; do not silently exclude them as infeasible.
5. **Use simultaneous uncertainty for the claim.** A conservative sufficient result is that our risk upper bound is at most $h_0$, cost is at most $c_0$, and our utility lower bound exceeds the utility upper bounds of all comparators at no higher cost that have not been demonstrated infeasible in risk. Predefine a smallest meaningful utility margin and size the fresh seed count for that comparison. Pointwise significance after searching many cells is insufficient.
6. **Close the adaptive-attack problem.** E9 permits exact worst-case attacks. General policies require attacking the complete acquisition and deployment pipeline, with optimization certificates or bounds on residual attack gaps. Report feasibility and gaps. A weak attack can exaggerate both safety and utility and cannot support the frontier claim.

The hypothesis fails in a declared regime if a fairly tuned comparator reaches the same useful gain with no more risk and lower cost, or reaches higher gain with no more risk or cost. It also fails if the benefit disappears after risk calibration, if protection always retains the incumbent, or if a feasible adaptive attack defeats the claimed safety bound. Price, integrity, overlap, and population-shift sensitivities should delimit the result rather than be tuned away.

## What remains for the course project and for a paper

For the course, the defensible contribution is an audited comparison of protected and corruptible evidence under adaptive reward poisoning, with a clear deployment objective, explicit costs, and falsifying cases. The proposal should map directly to the rubric: the failing deployment decision; fixed safety, utility, and cost metrics; strong baselines and their source algorithms; the source-weight method and the frontier hypothesis above; white-box knowledge and edit permissions; falsification and trust risks; and measured compute plus a separately labeled estimate for future experiments. The current E9 simulation is CPU-only and took 7.152 seconds; this does not predict the cost of trained-policy attacks. Public code, retained outcomes, and local synthetic attacks provide reproducibility and an explicit ethical scope.

The remaining course experiment is a frozen frontier comparison with independently tuned comparators and fresh confirmation. Its key unresolved comparison is source weighting versus influence allocation in the low-influence-good geometry. The presentation should make the cost constraint intelligible before discussing weighting, and show one positive result together with the unrestricted-acquisition counterexample.

For a paper, extend the method to independently trained policies with variable importance coefficients and a nontrivial candidate library. Derive a simultaneous guarantee for selecting a policy better than the incumbent after adaptive poisoning and any data-dependent design choice. Demonstrate the tradeoff on fresh feature-based benchmarks, under several logging-overlap conditions, and compare established OPE and robust-estimation methods on the same decision metrics. Stress the integrity and population agreement of the reference source. A particularly useful theoretical contribution would characterize when a protected sample can reduce the cost of a useful safe decision, with an upper bound and a matching or informative impossibility bound under a clearly specified model class.

This is an achievable research direction, not an acceptance prediction. The immediate priority is to resolve one fair frontier comparison; broadening benchmarks should follow a clear method and claim.
