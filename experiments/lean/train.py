"""SGS pipeline configuration for the main Lean experiments."""

import json
import os
from pathlib import Path

from transformers import set_seed

from experiments.lean.data import DatasetType
from experiments.lean.generation.config import ConjecturerSetup
from experiments.lean.pipeline.config import PipelineConfig, StatementSelectionMode
from experiments.lean.pipeline.loop import run_pipeline
from experiments.lean.support import get_standard_training_config
from experiments.lean.support.experiment_utils import (
    get_deepseek_prover_v2_prover_config,
    get_deepseek_prover_v2_conjecturer_config,
    get_local_running_config,
)


def main():
    run = Path(os.environ["COMPASSPLAY_RUN"])
    saved = json.loads((run / "run.json").read_text())
    config, variant = saved["config"], saved["variant"]
    set_seed(saved["seed"])
    alignment = variant in ("cos", "cos_novelty")
    frozen = variant == "solve_only"
    checkpoint = run / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    # Continue only from complete iterations, retaining both accumulated pools.
    start = 0
    for iteration in range(config["iterations"]):
        previous = checkpoint / f"iteration_{iteration}"
        complete = [previous / f"prover_dataset_{iteration}.pkl",
                    previous / f"conjecturer_dataset_{iteration}.pkl",
                    previous / f"model_prover_iteration_{iteration}/config.json"]
        if not frozen:
            complete.append(previous / f"model_conjecturer_iteration_{iteration}/config.json")
        if not all(path.is_file() for path in complete):
            break
        start = iteration + 1
    prover = get_deepseek_prover_v2_prover_config()
    prover.model_name = config["model"]
    prover.penalize_double_lean = False
    prover.penalize_try = True
    prover.dapo_length_penalty = True
    prover.stp_length_penalty = False
    prover.penalize_long_proof_str_over_1000 = False
    conjecturer = get_deepseek_prover_v2_conjecturer_config(
        ConjecturerSetup.SEED_STATEMENT if alignment
        else ConjecturerSetup.TARGET_STATEMENT_ONLY_UNSOLVED
    )
    conjecturer.model_name = config["model"]
    prover_data = config["dataset"]
    conjecturer_data = str(Path(config["dataset"]).parent / "conjecturer_dataset.json")
    if start:
        previous = checkpoint / f"iteration_{start - 1}"
        prover.model_name = str(previous / f"model_prover_iteration_{start - 1}")
        if not frozen:
            conjecturer.model_name = str(previous / f"model_conjecturer_iteration_{start - 1}")
        prover_data = str(previous / f"prover_dataset_{start - 1}.pkl")
        conjecturer_data = str(previous / f"conjecturer_dataset_{start - 1}.pkl")

    guide_class, guide = None, None
    if variant == "sgs":
        from experiments.lean.generation.guide import GuideConfig
        guide_prover = get_deepseek_prover_v2_prover_config()
        guide_prover.model_name = config["model"]
        guide = GuideConfig(prover_config=guide_prover, batch_size=1,
                            guide_model_path=config["guide_model"])
        guide_class = "DeepseekProverV2LemmaGuideLocal"
    training = get_standard_training_config()
    training.learning_rate = training.prover_learning_rate = training.conjecturer_learning_rate = 3e-6
    training.batch_size = 32
    training.reset_trainer = True
    resources = get_local_running_config()
    resources.log_dir = str(run / "jobs")
    pipeline = PipelineConfig(
        save_data_extension="pkl", eval_init_model=False, eval_datasets=[],
        num_master_verification_workers=110, pipeline_proving_and_verification=True,
        eval_in_seperate_job=False,
        prover_dataset_path=prover_data, conjecturer_dataset_path=conjecturer_data,
        proofs_per_sample=8, checkpoint_dir=str(checkpoint), training_config=training,
        retrain_prover_from_scratch=False, retrain_conjecturer_from_scratch=False,
        prover_iterations_in_buffer=1, conjecturer_iterations_in_buffer=1,
        num_train_examples=None, conjecture_multiplier=None,
        prover_model_config=prover, conjecturer_model_config=conjecturer,
        conjectures_per_statement={DatasetType.D_3K: 1},
        guide_class=guide_class, guide_config=guide,
        verification_address="local", verifier_timeout=int(os.environ.get("SGS_VERIFIER_TIMEOUT", "500")),
        pipeline_class="PipelineRunnerStandard", end_iteration=config["iterations"],
        start_iteration=start, parameter_sharing=False, num_conjecturer_warmup_rounds=0,
        training_resources_config=resources, gen_resources_config=resources,
        guide_resources_config=resources, verification_resources_config=resources,
        eval_gen_resources_config=resources, eval_verification_resources_config=resources,
        eval_every=200, eval_best_of_n=8, wandb_project="compassplay",
        wandb_resume_id=False, wandb_tags=[variant], freeze_prover=False,
        freeze_conjecturer=frozen, statement_selection_mode=StatementSelectionMode.HARD,
    )
    run_pipeline(pipeline_config=pipeline)


if __name__ == "__main__":
    main()
