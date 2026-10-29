# CompassPlay

This is the code repository for **CompassPlay: Rewarding the Proposer for Where It Moves the Solver**.

CompassPlay rewards generated tasks using the cosine between the solver’s loss gradient on a verified solution and a reference gradient. Lean experiments additionally weight this reward by statement novelty.

```text
method/                  shared method core
experiments/coding/      coding training and evaluation
experiments/lean/        Lean training and target coverage
```

## Setup

Run commands from this directory, using separate Python environments for Coding and Lean. Both require compatible AMD ROCm builds of PyTorch and vLLM already installed; setup scripts install additional dependencies. The paper uses eight MI355X GPUs for Coding and eight MI250X GPUs for Lean.

## Coding

Uses Qwen2.5-Coder-7B with shared proposer and solver weights, based on [AZR](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner) and [verl](https://github.com/volcengine/verl).

```bash
export MODEL=/path/to/Qwen2.5-Coder-7B
bash experiments/coding/scripts/setup.sh
bash experiments/coding/scripts/prepare_data.sh
bash experiments/coding/scripts/train.sh
```

The default is CompassPlay with 200 training steps on eight GPUs. Use `--variant difficulty` or `--variant solve_only` for baselines. Configure training with `TOTAL_STEPS`, `N_GPUS`, `TP`, or appended Hydra overrides.

Merge a saved checkpoint and evaluate all eight benchmarks:

```bash
bash experiments/coding/scripts/merge_checkpoint.sh /path/to/global_step_200/actor /path/to/hf_model
MODEL=/path/to/hf_model bash experiments/coding/scripts/evaluate.sh
```

Evaluation uses greedy decoding on LiveCodeBench execution, CRUXEval-I/O, MATH500, AMC23, GSM8K, Minerva Math, and OlympiadBench. Checkpoints are saved every five steps.

## Lean

Uses separate DeepSeek-Prover-V2-7B proposer and solver models, based on [SGS](https://github.com/LukeBailey181/sgs).

```bash
export MODEL=/path/to/DeepSeek-Prover-V2-7B
bash experiments/lean/scripts/setup.sh
bash experiments/lean/scripts/prepare_data.sh --preseed
bash experiments/lean/scripts/train.sh
```

The default is cosine × novelty for 150 iterations. Other variants are `cos`, `difficulty`, `sgs`, and `solve_only`, selected with `--variant`. The `sgs` variant accepts `GUIDE_MODEL=/path/to/dspv2_guide`. Set `TOTAL_ITERATIONS` to change run length.

Cosine variants condition on solved seeds and their proofs; other baselines condition on unsolved targets.

Each run writes cumulative coverage over 3,323 target theorems to `coverage.json` after each iteration’s verification.

## Runs

Data are downloaded by the preparation scripts into `datasets/`, dependencies in `upstream/`, and outputs in `runs/`. These directories are ignored by Git. Each new run automatically generates and records a random seed; resume retains it.

Both training scripts support `--dry-run` and `--resume /path/to/run`. Resume requires the same variant, model, and settings.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pu2026compassplay,
  title={CompassPlay: Rewarding the Proposer for Where It Moves the Solver},
  author={Pu, Sophia Xiao and Sun, Ximeng and Liu, Jiang and Wu, Jialian and Barsoum, Emad and Liu, Zicheng and Wang, William Yang},
  journal={arXiv preprint arXiv:2609.32228},
  year={2026}
}
```
