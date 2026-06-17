# SGCRL Torch Reproduction Study Knowledge Base

This document summarizes the repository state, implemented reproduction scope,
additional experiments, known limitations, and possible extensions. It is meant
as a handoff document for a new LLM instance that needs to write an article
about this reproduction study.

## Paper And Goal

This repository reproduces core components of the paper "A Single Goal is All
You Need: Skills and Exploration Emerge from Contrastive RL without Rewards,
Demonstrations or Subgoals". The study focuses on goal-conditioned
contrastive reinforcement learning, where a single fixed goal and future-state
relabeling can produce exploratory behavior and useful skills without external
reward engineering.

The main reproduction question is whether a PyTorch implementation can match
the high-level behavior of the original SGCRL codebase while avoiding the
original JAX/Acme/Reverb/Launchpad stack. The repository also contains
additional evaluation-only analyses designed to probe whether the learned
representations are behaviorally meaningful, not just whether policies achieve
success on benchmark rollouts.

## Repository Implementation

The codebase is a self-contained PyTorch reimplementation of SGCRL. It replaces
JAX, Haiku, Optax, Acme, Reverb, and Launchpad with:

- `torch` modules and optimizers.
- `torch.multiprocessing` for asynchronous actors.
- A local episode replay buffer with future-goal relabeling.
- CSV and terminal logging through a lightweight local logger.

The main package is `sgcrl_torch/`.

Important files:

- `train.py`: main training entry point.
- `sgcrl_torch/config.py`: dataclass config and algorithm flag handling.
- `sgcrl_torch/networks.py`: policy network and product-form contrastive critic.
- `sgcrl_torch/learner.py`: policy, critic, entropy, and target-network updates.
- `sgcrl_torch/runner.py`: actor/learner runtime, replay, logging, checkpointing.
- `sgcrl_torch/replay.py`: episode replay and future-goal relabeling.
- `sgcrl_torch/actors.py`: asynchronous actor loops and evaluation rollouts.
- `sgcrl_torch/env_utils.py`: point and Meta-World Sawyer environment wrappers.
- `sgcrl_torch/point_env.py`: 2D point-maze environments.
- `render_sawyer_box.py`: policy rollout rendering for Sawyer environments.
- `norm_visualizer.py`: Appendix-style visualization of goal-encoder norm.
- `critic_arch_ablation.py`: product critic versus monolithic critic experiment.
- `state_action_repr_geometry.py`: evaluation-only state-action representation analysis.

Supported algorithms:

- `contrastive_nce`
- `contrastive_cpc`
- `c_learning`
- `nce+c_learning`

Supported environments:

- `point_Spiral11x11`
- `sawyer_bin`
- `sawyer_box`
- `sawyer_peg`

## Training Runtime

Training uses asynchronous actor processes and a learner in the main process.
Actors collect full episodes and push them to a multiprocessing queue. The
learner drains completed episodes into a local replay buffer, samples
future-relabelled transitions, and performs many SGD updates per learner step.
Policy weights are periodically copied from learner to actors.

The default checkpoint path pattern is:

```text
logs/<alg>_<env>_<seed>/checkpoints/
```

The default learner log path is:

```text
logs/<alg>_<env>_<seed>/learner.csv
```

Logged training metrics include critic loss, actor loss, contrastive
classification metrics, entropy, actor step counts, replay counts, and
environment success metrics. If `--eval_every` is enabled, deterministic
evaluation metrics are also logged.

## Critic Architecture

The standard SGCRL critic is implemented as a product of two learned
representations:

```text
q(s, a, g) = phi(s, a)^T psi(g)
```

In code, this is `ContrastiveQNetwork` in `sgcrl_torch/networks.py`. It computes
all state-action/goal scores in a batch through an inner product matrix:

```text
logits[i, j] = phi(s_i, a_i)^T psi(g_j)
```

This structure is central to the paper's claim that useful representations and
skills emerge from contrastive RL.

## Warm Start Support

Both `train.py` and `critic_arch_ablation.py` support warm-starting from a
learner checkpoint:

```bash
python train.py \
  --env sawyer_box \
  --seed 0 \
  --warm_start_checkpoint logs/contrastive_cpc_sawyer_box_0/checkpoints/learner_10000.pt
```

Warm start loads:

- policy parameters
- critic parameters
- target critic parameters
- policy optimizer state
- critic optimizer state
- `num_sgd_steps`
- adaptive entropy state when relevant

It does not restore:

- replay buffer
- actor step counters
- logger state
- RNG states
- in-progress environment state

Therefore warm start is not an exact resume. It is a continuation from saved
weights and optimizers with fresh replay and fresh actors.

If warm-starting from a numbered checkpoint such as `learner_10000.pt`, future
numbered checkpoint names are offset by that number. For example, if the resumed
run locally reaches `learner_100.pt`, the saved file becomes:

```text
learner_10100.pt
```

`learner_final.pt` is not offset.

## Slurm Usage

The repository is intended to run on a Slurm node. The relevant conda
environment is:

```text
/megaverse/storage/giusti/.conda/envs/sgcrl
```

The current `train.slurm` runs `sawyer_box` across five seeds using a Slurm
array:

```bash
#SBATCH --array=0-4
SEED=${SLURM_ARRAY_TASK_ID}
python train.py --env sawyer_box --seed "${SEED}" --num_steps 15000000 --checkpoint_every 10000
```

There is also `monolithic.slurm` for the critic architecture ablation across
product and monolithic critic variants.

## Product Critic Versus Monolithic Critic Ablation

The script `critic_arch_ablation.py` implements the paper's representation
ablation from Section 4.3 / Figure 11. It compares:

```text
product critic:    q(s, a, g) = phi(s, a)^T psi(g)
monolithic critic: q(s, a, g) = f([s, a, g])
```

The monolithic critic scores every state-action/goal pair by concatenating
`s`, `a`, and `g` and passing them through an MLP. It is defined inside the
script so the core package remains unchanged.

The paper notes that this ablation used batch size 32 due to memory overhead
from the monolithic all-pairs critic. The script follows this with default:

```text
--batch_size 32
```

Example:

```bash
python critic_arch_ablation.py \
  --env sawyer_bin \
  --seeds 0 1 2 3 4 \
  --variants product monolithic \
  --eval_every 1000 \
  --eval_episodes 10
```

Logs are separated by critic variant:

```text
logs/critic_arch_ablation/contrastive_cpc_product_<env>_<seed>/
logs/critic_arch_ablation/contrastive_cpc_monolithic_<env>_<seed>/
```

Important comparison metrics in `learner.csv`:

- `environment_success_rate`
- `environment_success_1000`
- `environment_latest_success`
- `eval/success`
- `eval/success_1000`

`eval/success` is the cleaner metric when `--eval_every` is enabled.

## State-Action Representation Geometry Experiment

The script `state_action_repr_geometry.py` implements an evaluation-only
experiment to assess whether the learned state-action representation
`phi(s,a)` groups actions by behavioral effect.

It samples state-action pairs, computes:

```text
representation = phi(s, a)
effect = next_achieved_state - achieved_state
```

Then it checks whether nearest neighbors in representation space have similar
transition effects.

For `point_*` environments:

- Samples states from free maze cells.
- Samples evenly spaced action directions.
- Disables action noise.
- Computes deterministic one-step transition effects.

For Sawyer environments:

- Samples real one-step transitions using random actions.
- Computes effects in achieved-goal coordinates.

The script requires a product/representation critic checkpoint. It rejects
monolithic critic checkpoints because they do not expose `phi(s,a)`.

Example:

```bash
python state_action_repr_geometry.py \
  --checkpoint_path logs/contrastive_cpc_sawyer_bin_42/checkpoints/learner_190000.pt \
  --output_dir renders/state_action_repr_geometry \
  --num_samples 4096 \
  --nearest_k 10
```

Outputs:

- `*_sa_repr_pca_by_action.png`
- `*_sa_repr_pca_by_effect_angle.png`
- `*_sa_repr_pca_by_effect_speed.png`
- `*_nn_effect_cosine_hist.png`
- `*_repr_vs_effect_distance.png`
- `*_metrics.json`
- `*_data.npz`

Interpretation of these plots:

- PCA by action angle shows whether `phi(s,a)` encodes raw action direction.
- PCA by effect angle shows whether `phi(s,a)` encodes behavioral movement direction.
- PCA by effect speed shows whether it separates strong moves from weak/no-op moves.
- Nearest-neighbor effect cosine histogram compares representation-nearest pairs to random pairs. This is the strongest diagnostic.
- Representation distance versus effect distance tests whether the representation is globally metric-aligned with transition effects.

For the generated `learner_190000` Sawyer bin results:

```text
checkpoint: logs/contrastive_cpc_sawyer_bin_42/checkpoints/learner_190000.pt
env: sawyer_bin
num_samples: 4096
repr_dim: 64
mean_nn_effect_cosine: 0.2164
mean_random_effect_cosine: 0.0159
mean_nn_effect_distance: 0.0119
mean_random_effect_distance: 0.0144
mean_nn_action_cosine: 0.2906
mean_random_action_cosine: 0.0011
```

Interpretation:

- Representation-nearest pairs have more similar transition effects than random pairs.
- The effect is positive but moderate, not clean or decisive.
- PCA plots show a low-dimensional structure, but action and effect colors are mixed rather than cleanly separated.
- Global representation distance has near-zero correlation with transition-effect distance.
- Conclusion: `phi(s,a)` has learned local behavioral clustering, but not a clean global skill manifold.

## Goal Encoder Norm Visualization

The script `norm_visualizer.py` reproduces an Appendix-style diagnostic for the
goal encoder. It samples many candidate goal states and plots:

```text
||psi(g)||_2^2
```

This can reveal whether the goal encoder norm concentrates around important or
reachable regions. Existing outputs are stored in:

```text
renders/norm_visualizer/
```

The visualizer supports Sawyer bin, box, and peg with environment-specific
workspace bounds and synthetic goal-state construction.

## Rendering Policy Rollouts

`render_sawyer_box.py` renders saved Sawyer policy rollouts and can save GIFs
or MP4s. It supports camera selection and multiple rendering backends.

Existing rollout renders are stored under:

```text
renders/sawyer_bin_renders/
renders/sawyer_box_renders/
renders/sawyer_peg_renders/
```

These are useful for qualitative analysis in an article: they can show whether
the policy produces coherent object manipulation behavior beyond scalar success
metrics.

## Existing Results And Artifacts

Observed repository outputs include:

- Multiple Sawyer and point checkpoints under `logs/`.
- Sawyer rollout GIFs in `renders/sawyer_*_renders/`.
- Goal encoder norm plots in `renders/norm_visualizer/`.
- State-action representation geometry plots in
  `renders/state_action_repr_geometry/`.

The state-action geometry directory includes generated analyses for:

- `learner_97500`
- `learner_190000`
- `learner_final`

These are especially useful for an article section about whether learned
representations have meaningful behavioral geometry.

## What Has Been Validated

The following validations were run during development:

- `train.py --help` and syntax checks.
- `critic_arch_ablation.py --help` and syntax checks.
- `state_action_repr_geometry.py --help` and syntax checks.
- Synthetic warm-start smoke tests for `train.py`.
- Synthetic warm-start smoke tests for `critic_arch_ablation.py`.
- Checkpoint filename offset smoke tests: `learner_10000.pt` plus local
  `learner_100.pt` writes `learner_10100.pt`.
- State-action representation geometry smoke test on a point checkpoint,
  producing all expected visualization files.

## Known Limitations

The implementation is a pragmatic PyTorch reproduction, not a bit-for-bit
replica of the original code.

Known limitations:

- Warm start is not exact resume because replay, counters, logger state, and RNG
  state are not restored.
- Monolithic critic support is implemented in the ablation script rather than
  in the core package.
- State-action geometry analysis for Sawyer uses random one-step transitions;
  many actions produce very small effects, making the signal noisy.
- PCA plots can hide high-dimensional structure. Negative visual evidence in
  PCA should be interpreted cautiously.
- Existing representation analyses are evaluation-only and do not prove causal
  usefulness of the representation.
- There is no automatic "latest checkpoint" resolution for warm starts. The
  checkpoint path must be passed explicitly.

## Suggested Article Structure

A useful article about this reproduction study could follow this outline:

1. Motivation: SGCRL claims skills and exploration can emerge from a single
   goal through contrastive RL.
2. Reimplementation: describe the PyTorch actor/learner/replay architecture and
   how it mirrors the original distributed design.
3. Main training results: report success metrics across environments and seeds.
4. Critic architecture ablation: compare product critic against monolithic
   critic, following the paper's Section 4.3 / Figure 11.
5. Representation diagnostics: use goal encoder norm and state-action geometry
   plots to test whether learned representations are meaningful.
6. Qualitative rollouts: include rendered Sawyer behaviors.
7. Limitations: discuss warm-start versus exact resume, noisy Sawyer effect
   measurement, and non-identical implementation details.
8. Extensions: propose stronger evaluation-only tests and possible training
   ablations.

## Evaluation-Only Extensions

The user prefers extensions that evaluate already-trained models rather than
training new ones. Strong candidates:

1. Dense goal-grid success maps for point environments.
2. Unseen start-goal pair evaluation.
3. Critic reachability AUC: test whether `q(s,a,g)` predicts empirical future
   goal reachability.
4. Goal representation geometry: compare `psi(g)` distances to physical or
   geodesic distances.
5. State-action representation geometry: already implemented.
6. Goal perturbation sensitivity: smoothly interpolate goals and test whether
   behavior changes smoothly.
7. Counterfactual goal relabel evaluation on fixed trajectories.
8. Robustness to observation noise, action noise, and goal noise.
9. Goal ablations at evaluation: zero goals, shuffled goals, fixed goals.
10. Compare product and monolithic critic checkpoints using identical
    evaluation-only diagnostics.

## Training-Based Extensions

If additional training becomes acceptable, useful extensions include:

- More seeds for every environment and variant.
- Exact paper hyperparameter matching and sensitivity analysis.
- Batch-size ablation for product and monolithic critics.
- Representation dimension ablation.
- `repr_norm` ablation.
- Different goal relabeling distributions.
- Random-goal ratio ablation through `--random_goals`.
- Exact resume support with replay, counters, and RNG restoration.
- Core-package integration of monolithic critic instead of script-level patching.

## Key Claims To Evaluate In The Article

The article should separate three levels of evidence:

1. Performance evidence: do policies reach goals?
2. Ablation evidence: does the product representation critic outperform or
   learn differently from the monolithic critic?
3. Mechanistic evidence: do `phi(s,a)` and `psi(g)` have interpretable geometry
   connected to skills, reachability, or transition effects?

The current repository supports all three, but the strongest existing
mechanistic evidence is moderate rather than definitive. For example,
`learner_190000` shows nearest-neighbor effect similarity above random, but no
clear global metric alignment. That nuance should be preserved in the article.
