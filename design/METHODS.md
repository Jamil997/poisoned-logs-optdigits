# Estimation, acquisition, and corruption certificates

## Target and information boundary

There are N=1,797 fixed evaluation contexts. The target context distribution is P(x)=1/N. Frozen stochastic policies are the incumbent pi_0 and six candidates pi_j. Each evaluated library contains three of those candidates, exactly as specified in E3b. Write delta_j(x,a)=pi_j(a|x)-pi_0(a|x). The target improvement is

$$\Delta_j=\sum_{x,a}P(x)\delta_j(x,a)m(x,a),\qquad m(x,a)=\mathbb E[R\mid x,a].$$

The simulator has deterministic classification reward m(x,a)=1{a=y_x}. Only the environment and evaluator access y_x. The collector and confidence calculation receive policy probabilities and known context sampling probabilities. This is a randomized contextual-bandit experiment with active access to an unlabeled finite pool, not sequential reinforcement learning or a confounded observational-data experiment. Sampling changes the evidence distribution; the target population remains uniform.

## Joint experimental design

The original joint distribution is Q_0(x,a)=P(x)[0.8 pi_0(a|x)+0.02]. Define

$$w(x,a)=P(x)\max_{j=1,\ldots,6}|\delta_j(x,a)|,\quad W=\sum_{x,a}w(x,a),\quad Q_*(x,a)=w(x,a)/W.$$

All six candidates determine the collection design, even when evaluating a three-candidate library. For lambda in {0,.25,.5,.75,1}, collect iid pairs from

$$Q_\lambda=(1-\lambda)Q_0+\lambda[0.99Q_*+0.01Q_0].$$

No outcomes enter this construction. The support mixture ensures strictly positive probabilities. Original and action-only controls keep the context distribution uniform. The action-only design normalizes max_j|delta_j(x,a)| separately within each x and adds a 1% uniform-action mixture. The joint RMS comparator replaces the maximum by sqrt(mean_j delta_j(x,a)^2), then applies the same support mixture with Q_0.

### Why this reduces the influence of one edit

For any joint distribution Q, the coefficient of an observed reward in the target advantage estimator is

$$c_j(x,a)=\frac{P(x)\delta_j(x,a)}{Q(x,a)}.$$

The largest absolute coefficient across candidates and pairs is C(Q)=max_(x,a) w(x,a)/Q(x,a). Since w(x,a)<=C(Q)Q(x,a), summing gives W<=C(Q). Equality holds at Q_*, on the nonzero-disagreement support. Therefore Q_* solves the elementary minimax coefficient problem

$$\min_Q\max_{j,x,a}\frac{P(x)|\delta_j(x,a)|}{Q(x,a)}=W.$$

This optimizes a coefficient bound, **not** the final safety–utility–cost frontier. With the support mixture, C(Q_lambda)<=W/(.99 lambda) for lambda>0. On the frozen six-candidate pool, the actual maxima are 15.756 for Q_0 and 0.284 for Q_1.

The RMS comparator instead minimizes the mean second moment before support mixing: writing A(x,a)=P(x)^2 mean_j delta_j(x,a)^2, Cauchy–Schwarz gives min_Q sum A/Q=(sum sqrt(A))^2, attained by Q proportional to sqrt(A). Its strong empirical performance prevents attributing the result uniquely to maximum-influence optimization.

## Centered advantage estimator and sampling allowance

For an iid log (X_i,A_i,R_i) drawn under Q, use

$$\widehat\Delta_j=\frac1n\sum_i c_j(X_i,A_i)(R_i-1/2).$$

This is the paired DR estimator with constant reward model 1/2. It is unbiased because sum_a delta_j(x,a)=0 for every x. No consistency of a fitted reward model is required. Context importance weighting is essential: using delta_j/mu alone after changing context frequencies would estimate the wrong population.

For Z_j=c_j(X,A)(R-1/2), a label-free range length and second-moment bound are

$$B_j=\max_{x,a}|c_j(x,a)|,\qquad v_j=\frac14\sum_{x,a}Q(x,a)c_j(x,a)^2.$$

For binary clean rewards, E[Z_j^2]=v_j exactly; for rewards in [0,1], it is an upper bound. Variance is no larger than this second moment. Set t=log(M/alpha), M=3 and alpha=.05. Inverting the one-sided Bernstein tail yields

$$b_j=\frac{B_jt}{3n},\qquad \beta_j=b_j+\sqrt{b_j^2+\frac{2v_jt}{n}}.$$

With probability at least 1-alpha over the clean log, all M clean scores S_j=hat Delta_j-beta_j lower-bound their corresponding improvements. This confidence statement concerns a fixed design/library. It does not automatically provide simultaneous coverage over every reported experimental cell.

## Attacker, corrected gate, and deployment

The attacker knows the log, collection distribution, policies, rule, and true candidate ordering. It can replace at most k=floor(epsilon n) rewards with arbitrary values in [0,1]. Contexts, actions, propensities, policies, and the edit cap are trusted. Its goal is to minimize the true improvement of the selected policy; retaining the incumbent has utility zero.

For observed possibly poisoned rewards r, define the damage of removing a reward's favorable contribution:

$$d_{ji}(r_i)=c_{ji}r_i-\min(c_{ji},0),\qquad T_k(d_j)=\text{sum of the k largest entries of }d_j.$$

The corrected lower score is

$$L_j(r)=\frac1n\sum_i c_{ji}(r_i-1/2)-\beta_j-\frac1nT_k(d_j(r)).$$

This is the exact minimum of the affine score over at most k arbitrary reward replacements around the observed log. Since the original clean log is in that set, L_j(poisoned)<=S_j(clean). The clean-log simultaneous confidence event therefore protects the deployment rule against any adaptive allowed attack. Select the largest L_j only if it exceeds 10^-9; otherwise retain pi_0. The nominal comparator omits T_k/n.

## All-real-edit certificates and adaptive attack search

Let S_j(r) be the nominal score of the underlying clean log. With attacker budget k_a and declared correction budget k_d, every attacked corrected score is bounded below by

$$\ell_j=S_j(r)-\frac1n T_{k_a+k_d}(d_j(r)).$$

For k_d>=k_a, an upper bound is

$$u_j=S_j(r)-\frac1n T_{k_d-k_a}(d_j(r)).$$

For k_d<k_a, replace the subtraction by +(1/n)T_(k_a-k_d)(max(c_j,0)-c_j r). Budgets are capped by n in top-sum operations. These bounds permit arbitrary fractional replacement values, not only binary flips.

A candidate can only win if its upper bound exceeds the deployment threshold and every competitor's lower bound. Retention is possible only if all lower bounds are nonpositive (with numerical tolerance). These tests produce a superset C of feasible outcomes. Using evaluator-only true improvements gives

$$U_{\rm floor}=\min_{j\in C}\Delta_j,\quad \Delta_{\rm retain}=0.$$

A positive floor proves useful deployment under every allowed edit for that stored log. This is an evaluation certificate; the defender does not get the evaluator's labels or true improvements. A feasible attack supplies U_attained>=U_worst>=U_floor. A positive gap is retained and reported.

The search uses targeted top-k reward flips against target/rival margins and evaluates the actual final gate. Nominal gates additionally use an all-real L1 relaxation and checked LP dual bounds to exclude impossible outcomes. When initial envelopes permit harmful outcomes, the inherited E2 active-branch MILP attack is run with its original parameters. When all outcomes are already certified nonnegative, the confirmation avoids MILP restarts on safe alternatives and reports any remaining utility gap. The primary success criterion uses U_floor, so a failed or curtailed attack search cannot create a successful result by itself.

## Development clipping control

The first development stage clipped c_j to [-h,h] for h in {.25,.5,1,2,4,8}. It subtracted the valid worst-case bias

$$\eta_j=\frac12\sum_{x,a}|P(x)\delta_j(x,a)-Q(x,a)\operatorname{clip}(c_j,-h,h)|.$$

This follows by bounding the residual with |R-1/2|<=1/2. It retained the exact top-k correction and recomputed confidence terms for the clipped coefficient. These controls did not obtain a positive certified floor in that development grid. They are preserved rather than omitted from the research record; clipping was not selected for confirmation.

## Costs and interpretation

Let rho(x)=sum_a Q(x,a) and mu_Q(a|x)=Q(x,a)/rho(x). The reported per-query reward cost is

$$C_{\rm query}=\sum_x\rho(x)\sum_a[\pi_0(a\mid x)-\mu_Q(a\mid x)]m(x,a).$$

It compares actions on the **same sampled contexts**. It is distinct from subsequent deployment improvement under P. If contexts arrive from P, a rejection sampler with envelope R_max=max_x rho(x)/P(x) accepts with probability 1/R_max; expected arrivals per accepted query are R_max. Direct pool access avoids this arrival bottleneck but not necessarily labeling or intervention cost.

A simple illustrative net-reward calculation is H times mean deployment improvement minus n C_query, assuming a fixed post-study horizon H and incumbent actions on unqueried arrivals. The release does not claim a horizon-independent advantage over never experimenting. New datasets, realistic acquisition constraints, adaptive allocation, and stronger published acquisition baselines remain required for a paper.
