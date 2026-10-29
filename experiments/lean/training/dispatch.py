# Source: https://github.com/LukeBailey181/sgs/blob/64d2de468a0ddf222d5836248bea1d239a253f49/sgs/training/train.py; modified for CompassPlay.
from typing import List, Tuple, Optional, Dict
import logging
import time
import os
import tempfile
import pickle
import subprocess
import shutil

import torch
import wandb
from submitit.core.utils import FailedJobError, UncompletedJobError

from experiments.lean.generation.config import ModelConfig, ModelType, ResourcesConfig
from experiments.lean.training.config import TrainingConfig, TrainingSampleDatum
from experiments.lean.training.worker import finetune_local
from experiments.lean.support import SubmititCleanupExecutor

logger = logging.getLogger(__name__)


def finetune_model(
    group_size: int,
    trainer_cls: str,
    model_save_path: str,
    train_data: List[TrainingSampleDatum],
    training_config: TrainingConfig,
    resources_config: ResourcesConfig,
    model_config: ModelConfig,
    iteration: Optional[int] = None,
    val_data: Optional[List[Tuple[str, str]]] = None,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    wandb_log_prefix: str = "",
) -> None:
    """ 
    Finetune a model on the train data. This function basically serves
    as a switcher between different trainign paths and submit it or not.

    Args:
        model_save_path: Path to save the finetuned model
        train_data: List of tuples of (prompt, target, weight, log_probs)
        training_config: Training configuration
        resources_config: Resources configuration
        model_config: Model configuration
        val_data: Optional list of tuples of (prompt, target) for validation
    """
    result = []

    if len(train_data) == 0:
        print(f"No training data provided")
        print(f"Copying input model to the output path")
        if os.path.exists(model_config.model_name):
            shutil.copytree(model_config.model_name, model_save_path)
        else:
            print(f"Input model path does not exist: {model_config.model_name}")
            print(f"So we will just create an empty dir at the target")
            os.makedirs(model_save_path, exist_ok=True)

        return result

    if model_config.model_type == ModelType.LOCAL:
        if resources_config.submitit:
            num_retries = 3

            with SubmititCleanupExecutor(resources_config=resources_config) as executor:
                while num_retries > 0:
                    try:
                        job = executor.submit(
                            finetune_local,
                            group_size=group_size,
                            trainer_cls=trainer_cls,
                            model_save_path=model_save_path,
                            train_data=train_data,
                            training_config=training_config,
                            resources_config=resources_config,
                            model_config=model_config,
                            val_data=val_data,
                        )

                        # Wait for the job to finish
                        if resources_config.jobs_per_node > 1:
                            result: List[Dict[str, float]] = job.results()[0]
                        else:
                            result = job.result()  # type: ignore

                        # Training worked so break out of the loop
                        break

                    except (FailedJobError, UncompletedJobError) as e:
                        num_retries -= 1

                        logger.warning(
                            f"Training failed with error: {e}. "
                            f"Retries left: {num_retries}"
                        )

                        if num_retries == 0:
                            raise e
                        else:
                            # Sleep for 30 seconds before retrying
                            time.sleep(30)

        else:
            # Run locally via torchrun across all visible GPUs.
            num_gpus = torch.cuda.device_count()
            if num_gpus < 1:
                raise ValueError(
                    "Local training requires at least one visible CUDA device."
                )

            train_env = None
            _train_gpus = os.environ.get("SGS_TRAIN_GPUS", "").strip()
            if _train_gpus:
                _ids = [x.strip() for x in _train_gpus.split(",") if x.strip()]
                num_gpus = len(_ids)
                train_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(_ids)}
                # On ROCm, HIP_VISIBLE_DEVICES wins over CUDA_VISIBLE_DEVICES and
                # ROCR_VISIBLE_DEVICES wins over both. The generation phase leaves
                # HIP_VISIBLE_DEVICES set to the full list, so set both and clear ROCR.
                train_env["HIP_VISIBLE_DEVICES"] = ",".join(_ids)
                train_env.pop("ROCR_VISIBLE_DEVICES", None)
                logger.info(
                    f"SGS_TRAIN_GPUS set: training on {num_gpus} of "
                    f"{torch.cuda.device_count()} visible GPUs ({','.join(_ids)}); "
                    f"CUDA_/HIP_VISIBLE_DEVICES pinned, ROCR cleared"
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                args_path = os.path.join(tmpdir, "finetune_args.pkl")
                with open(args_path, "wb") as f:
                    # These are the arguments for finetune_local
                    pickle.dump(
                        {
                            "group_size": group_size,
                            "model_save_path": model_save_path,
                            "train_data": train_data,
                            "training_config": training_config,
                            "resources_config": resources_config,
                            "model_config": model_config,
                            "val_data": val_data,
                            "trainer_cls": trainer_cls,
                        },
                        f,
                    )
                # Pick a fresh free rendezvous port each launch. `--standalone`
                # reuses a fixed port -> EADDRINUSE when a prior iteration's socket
                # is still in TIME_WAIT (crashes repeated per-iteration torchrun).
                import socket as _sock
                _s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
                _s.bind(("127.0.0.1", 0)); _rdzv_port = _s.getsockname()[1]; _s.close()
                cmd = [
                    "torchrun",
                    "--nnodes=1",
                    f"--nproc_per_node={num_gpus}",
                    "--rdzv-backend=c10d",
                    f"--rdzv-endpoint=127.0.0.1:{_rdzv_port}",
                    "-m",
                    "experiments.lean.training.worker",
                    f"--args_pickle_path={args_path}",
                ]
                logger.info(
                    f"Launching local distributed training on {num_gpus} GPU(s): "
                    f"{' '.join(cmd)}"
                )
                _ = subprocess.run(cmd, check=True, env=train_env)

            # Collect metrics from trainer_state.json
            state_path = os.path.join(model_save_path, "log_history.pkl")
            result = []
            if os.path.exists(state_path):
                with open(state_path, "rb") as f:
                    result = pickle.load(f)

    # Now check if there is a checkpoint path in the load path and if so delete it 
    model_load_path = model_config.model_name
    if os.path.exists(model_load_path) and not training_config.reset_trainer:
        checkpoint_dirs = [x for x in os.listdir(model_load_path) if "checkpoint" in x]
        for checkpoint_dir in checkpoint_dirs:
            # check it is a dir and delete it
            if os.path.isdir(os.path.join(model_load_path, checkpoint_dir)):
                print(f"Deleting old checkpoint directory: {os.path.join(model_load_path, checkpoint_dir)}")
                shutil.rmtree(os.path.join(model_load_path, checkpoint_dir))


    if wandb_run is not None:
        for x in result:
            log_dict = {f"{wandb_log_prefix}/{k}": v for k, v in x.items()}

            if iteration is not None:
                log_dict["iteration"] = iteration

            wandb_run.log(log_dict)