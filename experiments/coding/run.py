"""Launch the three main coding experiments, or greedy AZR validation."""

import argparse
import os
from pathlib import Path
import sys

from experiments.runtime import ROOT, data_directory, input_identity, execute, new_run, require_files


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", choices=["cos", "difficulty", "solve_only"], default="cos")
    parser.add_argument("--resume", help="Existing run directory; reuse its automatic seed")
    parser.add_argument("--eval", action="store_true", help="Greedy CRUXEval-I/O and LiveCodeBench execution")
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    dataset = data_directory("coding")
    model = os.environ.get("MODEL", "Qwen/Qwen2.5-Coder-7B")
    steps = int(os.environ.get("TOTAL_STEPS", "200"))
    config = {"model": model, "steps": steps, "overrides": overrides,
              "gpus": int(os.environ.get("N_GPUS", "8")),
              "tp": int(os.environ.get("TP", "4" if args.eval else "1")),
              "max_token_len": 16384 if args.eval else 32768,
              "kl_coef": float(os.environ.get("KL_COEF", "0.01"))}
    ref_file = Path(os.environ.get("REF_FILE", dataset / "code_reason/gref_heldout_math.jsonl")).resolve()
    data_file = dataset / "code_reason/test_answer.parquet"
    seeds = [dataset / "7b_coder_seed_io.jsonl", dataset / "7b_coder_code_f_seed_io.jsonl"]
    if not args.dry_run:
        require_files(data_file, *seeds)
        if args.variant == "cos" and not args.eval:
            require_files(ref_file)
    inputs = [data_file, *seeds]
    if args.variant == "cos" and not args.eval:
        inputs.append(ref_file)
    config["inputs"] = input_identity(inputs)
    variant = "eval_coding" if args.eval else args.variant
    run, seed = new_run("coding", variant, config, args.resume)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ["RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"] = "1"
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.pop("ROCR_VISIBLE_DEVICES")
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])

    # Training settings used in the paper. AZR counts from step 1
    # and stops before total_training_steps, hence the +1.
    settings = {
        "trainer.project_name": "compassplay",
        "trainer.experiment_name": run.name,
        "trainer.default_local_dir": str(run / "checkpoints"),
        "trainer.n_gpus_per_node": config["gpus"], "trainer.nnodes": 1,
        "trainer.total_training_steps": 2 if args.eval else steps + 1,
        "trainer.save_freq": -1 if args.eval else 5, "trainer.test_freq": -1,
        "trainer.val_before_train": args.eval,
        "trainer.resume_mode": "disable" if args.eval else "auto",
        "trainer.logger": "[console]", "ray_kwargs.ray_init.num_cpus": 64,
        "data.train_files": str(data_file), "data.val_files": str(data_file),
        "data.train_batch_size": 64, "data.max_prompt_length": 6144,
        "data.max_response_length": 4096, "data.return_raw_chat": True,
        "data.seed": seed, "self_play.seed": seed,
        "actor_rollout_ref.model.path": model,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 192 if args.variant == "solve_only" else 384,
        "actor_rollout_ref.actor.data_loader_seed": seed,
        "actor_rollout_ref.actor.fsdp_config.seed": seed,
        "actor_rollout_ref.ref.fsdp_config.seed": seed,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 16,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": config["max_token_len"],
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.use_kl_loss": config["kl_coef"] != 0,
        "actor_rollout_ref.actor.kl_loss_coef": config["kl_coef"],
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": config["max_token_len"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": config["tp"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.5,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 16384,
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed": seed,
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class": "experiments.coding.workers.agent_loop.SelfPlayAgentLoopManager",
        "algorithm.adv_estimator": "reinforce_plus_plus",
        "reward_fn.extraction_type": "answer_conditional", "reward_fn.math_metric": "math_verify",
        "self_play.pretrained_tokenizer": True, "self_play.train_propose": args.variant != "solve_only",
        "self_play.seed_dataset": str(seeds[0]), "self_play.code_f_seed_dataset": str(seeds[1]),
        "self_play.output_seed_path": str(run / "seed_io.jsonl"),
        "self_play.output_code_f_seed_path": str(run / "code_f_seed_io.jsonl"),
        "self_play.problem_types": "[code_i,code_o,code_f]", "self_play.reward.n_samples": 8,
        "self_play.reward.code_f_reward_type": "binary",
        "self_play.data_selection_strategy.content_max_length": 5600,
        "self_play.data_selection_strategy.update_iteration": 1,
        "self_play.data_selection_strategy.seed_batch_factor": 4,
        "self_play.data_selection_strategy.valid_program_filter": "all",
        "self_play.data_selection_strategy.max_programs": 16384,
        "self_play.data_selection_strategy.composite_function_n_min": 0,
        "self_play.data_selection_strategy.composite_function_n_max": 0,
        "self_play.data_selection_strategy.composite_start_step": -1,
        "self_play.data_selection_strategy.composite_chance": 0.0,
        "self_play.data_selection_strategy.banned_keywords_for_errors_and_exceptions": "[raise]",
        "self_play.reward.generation_reward_config.use_original_code_as_ref": True,
        "self_play.reward.generation_reward_config.intrinsic_combine_method": "sum",
        "self_play.reward.generation_reward_config.gradient_alignment.enabled": args.variant == "cos" and not args.eval,
        "self_play.reward.generation_reward_config.gradient_alignment.coef": 1.0,
        "self_play.reward.generation_reward_config.gradient_alignment.max_response_length": 1152,
        "self_play.reward.generation_reward_config.gradient_alignment.ref_file": str(ref_file),
        "self_play.reward.generation_reward_config.gradient_alignment.n_ref": 32,
        "self_play.reward.generation_reward_config.gradient_alignment.ref_seed": 0,
    }
    if args.eval:
        settings["trainer.val_only"] = True
    def value(v):
        if isinstance(v, bool):
            return str(v).lower()
        # Quote paths for Hydra's grammar, including spaces and commas.
        if isinstance(v, str) and not v.startswith("["):
            import json
            return json.dumps(v)
        return str(v)
    command = [sys.executable, "-m", "experiments.coding.train"]
    command += [f"{key}={value(v)}" for key, v in settings.items()]
    command += overrides
    (run / "command.txt").write_text("\n".join(command) + "\n")
    execute(command, ROOT, args.dry_run)


if __name__ == "__main__":
    main()
