# Source: https://github.com/LeapLabTHU/Absolute-Zero-Reasoner/blob/484afa480c8f6fd77faa3d35451f24f287f58ee1/absolute_zero_reasoner/main_azr_ppo.py; modified for CompassPlay.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os
import ray
import hydra
from pathlib import Path
from pprint import pprint

from omegaconf import OmegaConf
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils import hf_tokenizer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

from experiments.coding.training.self_play import CodingSelfPlayTrainer
from experiments.coding.rewards.reward_managers import CodingRewardManager


@hydra.main(config_path='configs', config_name='train', version_base=None)
def main(config):
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        _compassplay_env = {k: os.environ[k] for k in os.environ if k.startswith("COMPASSPLAY_")}
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", **_compassplay_env}},
            num_cpus=config.ray_kwargs.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        from transformers import set_seed
        set_seed(config.self_play.seed)
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        if config.trainer.debug:
            import debugpy
            debugpy.listen(("0.0.0.0", config.trainer.debug_port))
            print(f"Debugger listening on port {config.trainer.debug_port}")
            debugpy.wait_for_client()
            print("Debugger attached!")

        # generator one batch, solver one batch
        config.actor_rollout_ref.actor.ppo_mini_batch_size = config.data.train_batch_size * len(config.self_play.problem_types) * (2 if config.self_play.train_propose else 1)
        pprint(f"auto setting ppo_mini_batch_size: {config.actor_rollout_ref.actor.ppo_mini_batch_size}")
        config.self_play.data_selection_strategy.data_len = config.data.train_batch_size * config.self_play.data_selection_strategy.update_iteration
        pprint(f"auto setting data_len: {config.self_play.data_selection_strategy.data_len}")

        # CompassPlay already allocates one unique directory per run.
        config.trainer.default_local_dir = Path(config.trainer.default_local_dir).as_posix()

        assert not (not config.self_play.reward.generation_reward_config.reject_multiple_functions and config.self_play.data_selection_strategy.composite_function_n_min > 0), "If reject_multiple_functions is False, composite_function_n_min must be 0"

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # base model chat template
        if config.self_play.pretrained_tokenizer:
            base_chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"
            tokenizer.chat_template = base_chat_template
            config.actor_rollout_ref.model.custom_chat_template = base_chat_template

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm.utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup

            from experiments.coding.workers.alignment import (
                AlignmentWorker,
            )

            actor_rollout_cls = AlignmentWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(
                f"only fsdp/fsdp2 are supported, got {config.actor_rollout_ref.actor.strategy}"
            )

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        assert config.algorithm.adv_estimator != "gae", (
            "no critic worker is registered; adv_estimator=gae is not supported"
        )
        assert not config.reward_model.enable, "reward model workers are not supported"
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = CodingRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            reward_fn_extraction_type=config.reward_fn.extraction_type,
            math_metric=config.reward_fn.math_metric,
            split='train',
            splitter=config.reward_fn.splitter,
            output_path=config.trainer.default_local_dir,
            max_prompt_length=config.data.max_prompt_length,
            generation_reward_config=config.self_play.reward.generation_reward_config,
            valid_program_filter=config.self_play.data_selection_strategy.valid_program_filter,
            debug=config.trainer.debug,
            extract_code_block=config.self_play.reward.extract_code_block,
            code_f_reward_type=config.self_play.reward.code_f_reward_type,
            boxed_retry=config.reward_fn.boxed_retry,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = CodingRewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            reward_fn_extraction_type=config.reward_fn.extraction_type,
            math_metric=config.reward_fn.math_metric,
            split='test',
            splitter=config.reward_fn.splitter,
            output_path=config.trainer.default_local_dir,
            max_prompt_length=config.data.max_prompt_length,
            generation_reward_config=config.self_play.reward.generation_reward_config,
            valid_program_filter=config.self_play.data_selection_strategy.valid_program_filter,
            debug=config.trainer.debug,
            extract_code_block=config.self_play.reward.extract_code_block,
            code_f_reward_type=config.self_play.reward.code_f_reward_type,
            boxed_retry=config.reward_fn.boxed_retry,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        wandb_tags = [
            'codeio', config.self_play.pred_data_mix_strategy, 'executor-' + config.self_play.executor,
            config.self_play.data_selection_strategy.valid_program_filter, config.self_play.gen_data_probabilities_strategy,
        ]
        wandb_tags.extend(config.self_play.problem_types)
        if config.trainer.wandb_tags is not None:
            config.trainer.wandb_tags = wandb_tags + config.trainer.wandb_tags.split(',')
        else:
            config.trainer.wandb_tags = wandb_tags

        trainer = CodingSelfPlayTrainer(
            past_epoch_window=config.self_play.past_epoch_window,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(0)
    except Exception as e:
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)
