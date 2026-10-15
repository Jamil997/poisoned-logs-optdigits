# Independent-policy reward-poisoning experiment

This document describes the E2 experiment. The original released-code reproduction is E0 and is preserved separately. E2 changes the benchmark design and does not claim to reproduce a table in the original paper.

## Question and estimand

Can a deployment rule retain useful improvement when an adversary can replace a bounded number of recorded rewards? The incumbent is a frozen stochastic classifier, and the six alternatives are also frozen before evaluation. A policy has continuous fitted parameters; the deployment decision is a choice among a finite library of trained policies.

For a context drawn uniformly from the official OptDigits test pool and a digit prediction drawn from a policy, define

$$
V(\pi)=\frac{1}{|T|}\sum_{x\in T}\pi(y_x\mid x),\qquad
\Delta_j=V(\pi_j)-V(\pi_0).
$$

The evaluator uses labels to calculate these finite-pool quantities exactly. This removes action-sampling noise from the outcome metric. It does not establish performance on an unknown population or across new fitted policy libraries.

The two primary axes are harmful-deployment probability and expected improvement:

$$
H=\Pr(\Delta_{\widehat j}<0),\qquad U=\mathbb E[\Delta_{\widehat j}],
$$

where fallback to the incumbent contributes zero to improvement. Reporting only harm would reward a rule that never deploys an alternative.

## Data separation

The official dataset contains 3,823 training examples and 1,797 test examples. A stratified split with seed 417 reserves 765 training examples for validation. Standardization and logistic-regression fitting use only the remaining 3,058 examples. The incumbent has C=0.1. The first three candidates use C in {0.01, 1, 10}; the remaining candidates mix the incumbent with the uniform policy using mixture weights {0.1, 0.25, 0.5}. The stochastic policy is the classifier's predicted probability vector, not its hard argmax prediction.

The reward model is selected from the same four regularization settings by validation log loss. The selected fit is retained without test-data refitting. Each method has access to the same clean training and validation information. Sharing this trusted training information is part of the threat model; it is not a setting in which all historical labels may be poisoned.

Evaluation samples contexts independently with replacement from the test pool. The logger is

$$
\mu(a\mid x)=(1-\gamma)\pi_0(a\mid x)+\gamma/10.
$$

The logged reward is one for a correct digit and zero otherwise. The logging probability is known because the experiment generates the action. No observational causal identification claim is made.

## Affine advantage estimators

Write d_j(x,a)=pi_j(a|x)-pi_0(a|x), and let q-hat be a fixed reward predictor. The paired doubly robust summand is

$$
Z_j(x,a,r)=\sum_b d_j(x,b)\widehat q(x,b)
+\frac{d_j(x,a)}{\mu(a\mid x)}\{r-\widehat q(x,a)\}.
$$

With the true logger, its expectation is Delta_j regardless of reward-model accuracy. The IPS variant uses q-hat=0. The centered DR variant uses q-hat=1/2. Fitted DR uses the validation-selected probability model. Subtracting two policy lower confidence bounds would not certify improvement; these estimators instead estimate the advantage directly.

The optimistic shrinkage baseline uses the weight mapping of Su et al. (2020):

$$
h_\lambda(w)=\frac{\lambda w}{w^2+\lambda},\qquad
c_j=h_\lambda(\pi_j/\mu)-h_\lambda(\pi_0/\mu).
$$

The direct-model term is unchanged. Lambda is chosen separately for each candidate, log size, and exploration level from {0.1, 1, 10, 100, 1000, infinity}. Selection minimizes exact paired-estimator MSE on the validation pool under the specified logger. This is **our supervised-validation tuning rule**, not a reproduction of Su et al.'s estimator-selection procedure.

Shrinkage introduces bias. Define e_j=mu*c_j-d_j. A label-free upper bound on upward bias is

$$
\beta_j=\frac1{|T|}\sum_{x\in T}\sum_a
\max\{-e_j(x,a)\widehat q(x,a),\ e_j(x,a)[1-\widehat q(x,a)]\}.
$$

We subtract this allowance before certifying improvement. A smaller point-estimation MSE does not necessarily produce a useful confidence bound after accounting for bias.

## Confidence terms

Each summand is z0_j(x,a)+c_j(x,a)*r. The known finite context pool and logger allow bounds computed from features and policy probabilities alone:

$$
v_j=\frac1{|T|}\sum_x\sum_a\mu(a\mid x)
\max\{z_{0j}(x,a)^2,[z_{0j}(x,a)+c_j(x,a)]^2\}.
$$

Let B_j be the global range of these two endpoint values, and t=log(6/alpha). Since Var(Z_j) <= E[Z_j^2] <= v_j, Bernstein's inequality gives the radius

$$
s_j=\frac{B_jt}{3n}+\sqrt{\left(\frac{B_jt}{3n}\right)^2+\frac{2v_jt}{n}}.
$$

The IPS/Hoeffding comparator uses B_j*sqrt(t/(2n)). Point-estimate IPS and DR set the radius to zero and have no statistical confidence guarantee. These are ablations, not certified alternatives.

This design assumes the context distribution is the known uniform finite pool. Test feature vectors are available for the range and moment calculations, but test labels are not. Generalizing to an unknown deployment context distribution would require an additional concentration or distributional-uncertainty argument.

## Exact reward-edit correction

For an observed reward vector r-tilde, define

$$
U_j(\widetilde D)=\frac1n\sum_i Z_{ij}(\widetilde r_i)-s_j-\beta_j,
\qquad
d_{ij}=c_{ij}\widetilde r_i-\min(0,c_{ij}).
$$

The maximum downward empirical change from replacing at most k_d rewards in [0,1] is

$$
C_{k_d,j}(\widetilde D)=\frac1n\sum_{i\in\mathrm{Top}_{k_d}(d_{\cdot j})}d_{ij},
\qquad L_j=U_j-C_{k_d,j}.
$$

Select the candidate with the largest L_j if that score exceeds 1e-9; otherwise retain the incumbent. Ties use the lowest candidate index. Nominal rules set k_d=0. Corrected rules use k_d equal to the attack budget even in the clean and random-corruption conditions. This keeps the defender fixed while changing the attack and exposes the clean-data cost of protection.

If the attacker changes at most k_a<=k_d rows, the clean reward vector is feasible inside the defender's correction set. Thus L_j(poisoned)<=U_j(clean). Combined with valid clean confidence bounds and the union bound across six fixed candidates, this gives the stated alpha-level harmful-selection control, conditional on the assumptions. It does not establish zero risk, and it does not apply to the point-estimate ablations.

## Attack and certificates

The attacker knows the policies, reward model, complete log, and final decision rule. For a strong evaluation, it also knows the true ordering of the candidate values. The defender never receives that ordering. The attack seeks the least valuable reachable deployment, counting fallback as zero improvement.

The search evaluates targeted top-k binary-flip attacks against each actual gate, then uses an active-branch mixed-integer search with competitor cutting planes. Each returned witness is evaluated through the final corrected decision rule. A successful attack is evidence of reachability; a failed or timed-out search is not evidence of robustness.

Independent envelopes cover **every real-valued replacement** within the budget. For k_d>=k_a,

$$
U_j(D)-C_{k_d+k_a,j}(D)\le L_j(\widetilde D)
\le U_j(D)-C_{k_d-k_a,j}(D).
$$

For nominal affine rules, an additional linear-program relaxation allows fractional flips z_i in [0,1] with sum(z_i)<=k_a. This set contains every allowed sparse real-valued attack. A checked dual upper bound below zero excludes a target. Saved dual multipliers allow the audit to recompute these certificates independently of solver status.

The intervals exclude a candidate if its upper score cannot beat another candidate's lower score or the fallback threshold. The minimum true advantage among remaining possible decisions is a certified worst-case utility floor. The attained attack utility is an upper bound on the minimum. Equality closes the optimization gap for the full real-valued threat model. Otherwise the unresolved interval is reported. Binary-search bounds alone are not presented as certificates for fractional attacks.

## ESLB supplement

The original ESLB class is loaded unchanged from the pinned authors' source. To compare against an incumbent, our adapter adds the candidate lower bound on reward to the incumbent lower bound on complementary reward, then subtracts one. Algebraically this compares a candidate lower bound with an incumbent upper bound. It uses alpha/7 for the seven policy bounds and Bernstein bias control, which permits general bounded rewards.

There are ten log seeds per setting, shared with the first ten main seeds. The primary supplement uses 10 Monte Carlo iterations with batch size 100. A separate stability check uses batch size 1,000 for the first seed of each setting. Both the smaller seed count and reduced primary Monte Carlo budget are declared in the protocol. Numerical integration error is not certified, and the supplement is not pooled with the 100-seed main comparisons.

## Sources and attribution

- [Dudik et al. (2011), Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601): the DR estimator. Our implementation uses a paired advantage and independent supervised fitting.
- [Su et al. (2020), Doubly robust off-policy evaluation with shrinkage](https://proceedings.mlr.press/v119/su20a.html): the optimistic weight mapping. Validation tuning, the paired bias allowance, and the safe-deployment gate here are adaptations.
- [Kuzborskij et al. (2021), Confident Off-Policy Evaluation and Selection through Self-Normalized Importance Weighting](https://proceedings.mlr.press/v130/kuzborskij21a.html): the original OptDigits reproduction and unmodified ESLB estimator. The independent-policy data split and baseline-relative security adapter are ours.
- [Lobo et al. (2022), Data poisoning attacks on off-policy policy evaluation methods](https://proceedings.mlr.press/v180/lobo22a.html): the established adversarial OPE problem and influence-function attack framework. We do not reproduce their fitted-estimator attacks; our attack targets a different, frozen-model reward-only decision rule.
- [Alpaydin and Kaynak, UCI Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49): public data, licensed CC BY 4.0. Only local benchmark copies are attacked.

The combination is an empirical research prototype. The estimator, concentration tools, and top-k optimization do not by themselves establish algorithmic novelty or Pareto-optimality. Synthetic diversity and reward-model-error ablations, additional datasets, a stronger policy-library design, and an unknown-context-distribution extension remain separate work.
