# SGCRL Torch

This folder is a self-contained PyTorch reimplementation of the original SGCRL
repository. It keeps the same high-level behavior:

- actors run environment loops asynchronously
- complete episodes are pushed to replay
- replay samples transitions and relabels goals from future states
- the learner owns all optimizers and performs many SGD updates per learner step
- actors periodically refresh stale policy weights from the learner
- supported algorithms are `contrastive_nce`, `contrastive_cpc`, `c_learning`, and `nce+c_learning`

The port intentionally avoids Acme, Reverb, Launchpad, JAX, Haiku, and Optax.
It replaces them with `torch`, `torch.multiprocessing`, a local episode replay
buffer, and lightweight CSV/terminal logging.

## Install

Use the same Mujoco/Metaworld setup as the original repo for Sawyer tasks. For
the point environment only, the lightweight dependencies are enough:

```bash
cd sgcrl-torch
pip install -r requirements.txt
```

## Run

```bash
python train.py --env=point_Spiral11x11 --alg=contrastive_cpc
```

Sawyer tasks use the original names:

```bash
python train.py --env=sawyer_bin --alg=contrastive_cpc
python train.py --env=sawyer_box --alg=c_learning
python train.py --env=sawyer_peg --alg=nce+c_learning
```

Useful flags mirror the original entrypoint:

- `--env`: `sawyer_bin`, `sawyer_box`, `sawyer_peg`, `point_Spiral11x11`
- `--alg`: `contrastive_nce`, `contrastive_cpc`, `c_learning`, `nce+c_learning`
- `--num_steps`: maximum actor environment steps
- `--sample_goals`: sample goals from the environment instead of using the fixed single goal
- `--add_uid`: add a unique id inside the run log directory

Torch-specific flags:

- `--device=auto|cpu|cuda`
- `--num_actors=4`
- `--batch_size=256`
- `--num_sgd_steps_per_step=64`
- `--actor_update_period=100`
- `--eval_every=N`

## How It Mimics The Original

The original code has separate Launchpad nodes for replay, learner, actors, and
evaluation. This port keeps the same separation of responsibilities:

- actor processes own environments and never call `backward()`
- the main learner process owns replay, policy optimizer, critic optimizer, target critic, and optional alpha optimizer
- actors push entire episodes through a multiprocessing queue
- replay samples future goals from the same episode with discounted future-state probabilities
- every learner iteration runs `num_sgd_steps_per_step` gradient updates
- policy weights are copied to actors through a shared state snapshot

Checkpoints are written to:

```text
logs/<alg>_<env>_<seed>/checkpoints/
```

CSV learner logs are written to:

```text
logs/<alg>_<env>_<seed>/learner.csv
```


## Render Sawyer Rollouts

`render_sawyer_box.py` can render saved Sawyer policy rollouts and already
supports choosing the MuJoCo/Metaworld camera through `--camera_name`. The
default is `corner`:

```bash
python render_sawyer_box.py --camera_name corner
```

To change the viewing angle, first list the camera names exposed by your
installed Metaworld/MuJoCo environment:

```bash
python render_sawyer_box.py --list_cameras
```

Then pass one of those names with `--camera_name`. Common Metaworld camera names
include `corner`, `corner2`, `corner3`, `topview`, `behindGripper`, and
`gripperPOV`, depending on the installed version:

```bash
python render_sawyer_box.py --camera_name topview
python render_sawyer_box.py --camera_name corner2 --episodes 3 --format mp4
python render_sawyer_box.py --camera_name behindGripper --width 800 --height 600
```

The default `auto` backend now prefers Gymnasium/MuJoCo's camera-aware renderer
when `--camera_name` is set. You can also force that backend explicitly:

```bash
python render_sawyer_box.py --camera_name topview --render_backend mujoco_renderer
```

Useful render options:

- `--output_dir renders/sawyer_box_final`: directory for generated videos
- `--episodes 5`: number of rollout videos to render
- `--format gif|mp4`: output video format
- `--width 640 --height 480`: video resolution
- `--display`: also call `env.render()` for an interactive viewer while recording
- `--no_video`: run the policy and print metrics without writing videos
- `--render_backend auto|wrapper|sim|mujoco_renderer`: choose the renderer used for saved frames

When camera names are discoverable, the script validates `--camera_name` and
prints the available names if the requested camera does not exist. If no frames
can be captured, the script raises an error instead of silently leaving an empty
output directory.

## Robustness And Self-Recovery Perturbations

`self_recovery_perturbation.py` reproduces the paper's "Further training:
agent develops robustness and self-recovery" evaluation for `sawyer_box` and
`sawyer_peg`. It loads one or more checkpoints, perturbs the manipulated object
by up to 5 cm either immediately after reset (`static`) or halfway through the
episode (`dynamic`), and writes per-episode and aggregate recovery metrics:

```bash
python self_recovery_perturbation.py \
  --checkpoints \
    logs/contrastive_cpc_sawyer_peg_42/checkpoints/learner_50000.pt \
    logs/contrastive_cpc_sawyer_peg_42/checkpoints/learner_final.pt \
  --labels early further \
  --settings static dynamic \
  --episodes 50 \
  --output_dir renders/sawyer_peg_self_recovery
```

Outputs are written to:

- `episodes.csv`: per-rollout success, distance, perturbation, and recovery-step metrics
- `summary.csv`: success and recovery aggregates by checkpoint and perturbation setting
- `summary.json`: the same aggregate metrics in JSON format

To inspect qualitative behavior, render the first few perturbed rollouts:

```bash
python self_recovery_perturbation.py \
  --checkpoints logs/contrastive_cpc_sawyer_box_42/checkpoints/learner_final.pt \
  --settings dynamic \
  --episodes 10 \
  --render_episodes 3 \
  --camera_name corner2
```
