# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
import os
from dataclasses import asdict
from enum import Enum
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.scheduler.cluster import Cluster
from rlinf.utils.placement import HybridComponentPlacement

logging.getLogger().setLevel(logging.INFO)


class SupportedModel(Enum):
    # Embodied models
    LEROBOT_PI05 = ("lerobot_pi05", "embodied")

    # Sft models
    QWEN2_5_VL_SFT = ("qwen2.5_vl", "sft")
    QWEN3_VL_SFT = ("qwen3_vl", "sft")
    QWEN3_VL_MOE_SFT = ("qwen3_vl_moe", "sft")

    def __new__(cls, value, category):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.category = category
        return obj


def get_supported_model(model_type: str) -> SupportedModel:
    try:
        return SupportedModel(model_type)
    except ValueError as err:
        supported_models = [e.value for e in SupportedModel]
        raise NotImplementedError(
            f"Model Type: {model_type} not supported. Supported models: {supported_models}"
        ) from err


SUPPORTED_TASK_TYPE = ["embodied"]
SUPPORTED_TRAINING_BACKENDS = ["fsdp"]
__all__ = ["build_config"]


def torch_dtype_from_precision(
    precision: Union[int, str, None],
) -> Optional[torch.dtype]:
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    elif precision in [32, "32", "fp32", "32-true"]:
        return torch.float32
    elif precision in [None, "null"]:
        return None
    else:
        raise ValueError(
            f"Could not parse the precision of `{precision}` to a valid torch.dtype"
        )


@torch.jit.script
def gelu_impl(x):
    """
    OpenAI's gelu implementation.
    """
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


def openai_gelu(x):
    return gelu_impl(x)


try:
    jit_fuser = torch.compile
except Exception:
    jit_fuser = torch.jit.script


@jit_fuser
def squared_relu(x):
    return torch.pow(torch.nn.functional.relu(x), 2)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)
            + torch.ones_like(x).to(dtype=x.dtype)
        )
    )


def activation_to_func(
    activation: str, openai_gelu: bool = False, onnx_safe: bool = False
) -> Callable:
    """
    Converts an activation function represented as a string to a function.

    Args:
        activation (str): string representation of an activation function, typically gotten from the model config.
        openai_gelu (bool): whether to use the OpenAI GELU implementation. Used with HF compatibility.
        onnx_safe (bool): whether to use the ONNX-compatible implementation of GELU.

    Returns:
        Callable: the activation function.
    """

    supported_activations = [
        "gelu",
        "geglu",
        "reglu",
        "swiglu",
        "squared-relu",
        "fast-geglu",
        "fast-swiglu",
        "fast-reglu",
        "approx-gelu",
    ]

    if activation not in supported_activations:
        raise ValueError(
            f"Unsupported activation {activation}. Supported activations: {supported_activations} "
        )

    # Give openai_gelu precedence over other activations if set, for HF compatibility.
    # Normally this is off and shouldn't affect regular model training.
    if openai_gelu:
        activation_func = openai_gelu
    elif activation in ["gelu", "geglu", "fast-geglu"]:
        activation_func = F.gelu
    elif onnx_safe:
        activation_func = erf_gelu
    elif activation in ["reglu", "fast-reglu"]:
        activation_func = F.relu
    elif activation in ["swiglu", "fast-swiglu"]:
        # SiLU or sigmoid linear unit is the same as swish with beta = 1 (which is what https://arxiv.org/pdf/2002.05202.pdf uses.)
        activation_func = F.silu
    elif activation == "squared-relu":
        activation_func = squared_relu

    return activation_func


def validate_model_cfg_by_hf_config(cfg, hf_model_path):
    # validate by hf config
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

    if "Qwen2ForCausalLM" in hf_config.architectures:
        qkv_bias = True
    else:
        qkv_bias = getattr(hf_config, "attention_bias", False)

    if (
        "Qwen3ForCausalLM" in hf_config.architectures
        or "Qwen3MoeForCausalLM" in hf_config.architectures
    ):
        qk_layernorm = True
    else:
        qk_layernorm = getattr(cfg.model, "qk_layernorm", False)

    with open_dict(cfg):
        rs = getattr(hf_config, "rope_scaling", None)
        if isinstance(rs, dict):
            rtype = rs.get("type", "")
            if rtype in {"linear", "dynamic", "ntk", "yarn"}:
                f = rs.get("factor")
                if f is not None:
                    cfg.model.seq_len_interpolation_factor = float(f)
            else:
                # mrope
                cfg.model.seq_len_interpolation_factor = None
        cfg.model.padded_vocab_size = hf_config.vocab_size
        cfg.model.max_position_embeddings = hf_config.max_position_embeddings
        cfg.model.rotary_base = hf_config.rope_theta
        cfg.model.share_embeddings_and_output_weights = getattr(
            hf_config, "tie_word_embeddings", False
        )
        cfg.model.num_layers = hf_config.num_hidden_layers
        cfg.model.hidden_size = hf_config.hidden_size
        cfg.model.num_attention_heads = hf_config.num_attention_heads
        cfg.model.num_query_groups = hf_config.num_key_value_heads
        cfg.model.ffn_hidden_size = hf_config.intermediate_size
        cfg.model.attention_dropout = hf_config.attention_dropout
        cfg.model.hidden_dropout = getattr(hf_config, "hidden_dropout", 0.0)
        cfg.model.add_qkv_bias = qkv_bias
        cfg.model.qk_layernorm = qk_layernorm
        cfg.model.layernorm_epsilon = hf_config.rms_norm_eps
        cfg.model.head_dim = getattr(
            hf_config,
            "head_dim",
            cfg.model.hidden_size // cfg.model.num_attention_heads,
        )
        if cfg.model.head_dim is not None:
            cfg.model.kv_channels = cfg.model.head_dim

        # MoE model
        cfg.model.num_moe_experts = getattr(hf_config, "num_experts", None)
        cfg.model.num_experts = getattr(hf_config, "num_experts", None)
        cfg.model.moe_ffn_hidden_size = getattr(
            hf_config, "moe_intermediate_size", None
        )
        cfg.model.moe_router_topk = getattr(hf_config, "num_experts_per_tok", 2)

    return cfg


def validate_fsdp_cfg(cfg: DictConfig) -> DictConfig:
    def validate_amp_cfg(config: DictConfig) -> DictConfig:
        """Validate AMP configuration and ensure mutual exclusivity with FSDP mixed_precision."""

        param_dtype = config.mixed_precision.param_dtype
        reduce_dtype = config.mixed_precision.reduce_dtype
        buffer_dtype = config.mixed_precision.buffer_dtype

        all_none = param_dtype is None and reduce_dtype is None and buffer_dtype is None

        all_fp32 = (
            param_dtype == "fp32" and reduce_dtype == "fp32" and buffer_dtype == "fp32"
        )

        use_fsdp_mixed_precision = not (all_none or all_fp32)

        amp_autocast = config.get("amp_autocast", {})
        config.amp_autocast = {
            "enabled": amp_autocast.get("enabled", False),
            "precision": amp_autocast.get("precision", "bf16"),
        }

        grad_scaler = config.get("grad_scaler", {})
        config.grad_scaler = {
            "enabled": grad_scaler.get("enabled", False),
            "init_scale": grad_scaler.get("init_scale", None),
            "growth_interval": grad_scaler.get("growth_interval", None),
        }

        if "amp" in config:
            logging.warning(
                "fsdp_config.amp is no longer supported, use fsdp_config.amp_autocast and fsdp_config.grad_scaler instead"
            )

        if config.amp_autocast.enabled and use_fsdp_mixed_precision:
            assert False, (
                "amp_autocast should not be enabled when fsdp mixed_precision is enabled"
            )
        assert config.amp_autocast.precision in ["fp16", "bf16", "fp32"], (
            "fsdp.amp_autocast.precision must be one of ['fp16', 'bf16', 'fp32']"
        )
        return config

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.fsdp_config.strategy = cfg.fsdp_config.get("strategy", "fsdp")

        cfg.fsdp_config.sharding_strategy = cfg.fsdp_config.get(
            "sharding_strategy", "full_shard"
        )

        cfg.fsdp_config.forward_prefetch = cfg.fsdp_config.get(
            "forward_prefetch", False
        )
        cfg.fsdp_config.limit_all_gathers = cfg.fsdp_config.get(
            "limit_all_gathers", False
        )
        cfg.fsdp_config.backward_prefetch = cfg.fsdp_config.get(
            "backward_prefetch", None
        )
        cfg.fsdp_config.use_orig_params = cfg.fsdp_config.get("use_orig_params", False)
        cfg.fsdp_config.use_liger_kernel = cfg.fsdp_config.get(
            "use_liger_kernel", False
        )

        cfg.fsdp_config.cpu_offload = cfg.fsdp_config.get("cpu_offload", False)
        cfg.fsdp_config.offload_pin_memory = cfg.fsdp_config.get(
            "offload_pin_memory", False
        )
        cfg.fsdp_config.reshard_after_forward = cfg.fsdp_config.get(
            "reshard_after_forward", True
        )
        cfg.fsdp_config.enable_gradient_accumulation = cfg.fsdp_config.get(
            "enable_gradient_accumulation", False
        )

        assert cfg.fsdp_config.backward_prefetch in [
            None,
            "pre",
            "post",
        ], "fsdp_config.backward_prefetch must be one of [None, 'pre', 'post']"

        # validate mixed precision config
        assert hasattr(cfg.fsdp_config, "mixed_precision"), (
            "fsdp_config.mixed_precision is required in FSDP actor configuration."
        )
        mixed_precision_config = cfg.fsdp_config.mixed_precision
        mixed_precision_config.param_dtype = mixed_precision_config.get(
            "param_dtype", None
        )
        mixed_precision_config.reduce_dtype = mixed_precision_config.get(
            "reduce_dtype", None
        )
        mixed_precision_config.buffer_dtype = mixed_precision_config.get(
            "buffer_dtype", None
        )
        cfg.fsdp_config = validate_amp_cfg(cfg.fsdp_config)

    return cfg


def validate_embodied_cfg(cfg):
    assert get_supported_model(cfg.actor.model.model_type).category == "embodied", (
        f"Model type: '{cfg.actor.model.model_type}' is not an embodied model. "
        f"Supported embodied models: {[e.value for e in SupportedModel if e.category == 'embodied']}."
    )

    # NOTE: Currently we only support actor_critic as PPO algorithm loss, and only support value_head as critic model.
    # This will be updated in the future to support more algorithms and critic models.
    # Check that actor_critic loss requires value_head
    if (
        cfg.algorithm.loss_type == "actor_critic"
        or cfg.algorithm.loss_type == "decoupled_actor_critic"
    ):
        add_value_head = cfg.actor.model.get("add_value_head", False)
        assert add_value_head, (
            f"When using PPO algorithm (algorithm.loss_type='actor_critic'), "
            f"actor.model.add_value_head must be True. "
            f"Current value: {add_value_head}"
        )

    # Warn if GAE is used without collect_prev_infos: preprocess_embodied_advantages_inputs
    # will receive values=None and crash when trying to reshape it.
    if cfg.algorithm.adv_type == "gae":
        collect_prev_infos = cfg.rollout.get("collect_prev_infos", True)
        if not collect_prev_infos:
            logging.warning(
                "algorithm.adv_type='gae' requires rollout.collect_prev_infos=True "
                "so that prev_values (used for GAE bootstrapping) are collected. "
                "Set rollout.collect_prev_infos: true in the config."
            )

    # Warn if the planner interval is larger than the number of chunk steps in
    # a rollout epoch: the planner would never fire because bootstrap_step()
    # resets the counter every rollout epoch.
    if cfg.get("marl", {}).get("enabled", False):
        subtask_interval = cfg.marl.get("planner", {}).get("interval", 0)
    else:
        subtask_interval = cfg.env.train.get("subtask_interval", 0)
    max_steps = cfg.env.train.get("max_steps_per_rollout_epoch", None)
    num_chunks = cfg.actor.model.get("num_action_chunks", None)
    if subtask_interval > 0 and max_steps is not None and num_chunks is not None:
        n_train_chunk_steps = max_steps // num_chunks
        if subtask_interval > n_train_chunk_steps:
            logging.warning(
                f"subtask_interval ({subtask_interval}) > n_train_chunk_steps "
                f"({n_train_chunk_steps}). The subtask planner will never fire "
                f"because bootstrap_step() resets the counter each rollout epoch. "
                f"Set subtask_interval <= {n_train_chunk_steps}."
            )

    # Warn if global_batch_size does not divide rollout_size.
    # rollout_size = n_train_chunk_steps * total_num_envs * rollout_epoch, and
    # run_training asserts rollout_size % (global_batch_size // world_size) == 0.
    # This check assumes world_size=1 at config-validate time; multi-GPU users should
    # ensure global_batch_size // world_size divides rollout_size.
    _max_steps = cfg.env.train.get("max_steps_per_rollout_epoch", None)
    _num_chunks = cfg.actor.model.get("num_action_chunks", None)
    _total_envs = cfg.env.train.get("total_num_envs", None)
    _rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
    _global_bs = cfg.actor.get("global_batch_size", None)
    if (
        _max_steps is not None
        and _num_chunks is not None
        and _total_envs is not None
        and _global_bs is not None
    ):
        _rollout_size = (_max_steps // _num_chunks) * _total_envs * _rollout_epoch
        if _rollout_size % _global_bs != 0:
            logging.warning(
                f"actor.global_batch_size ({_global_bs}) does not divide "
                f"rollout_size ({_rollout_size} = "
                f"n_train_chunk_steps={_max_steps // _num_chunks} * "
                f"total_num_envs={_total_envs} * rollout_epoch={_rollout_epoch}). "
                f"EmbodiedFSDPActor.run_training will assert-fail at the first "
                f"training step. Set actor.global_batch_size to a divisor of "
                f"{_rollout_size}, e.g. global_batch_size: {_rollout_size}."
            )

    # process num-envs
    component_placement = HybridComponentPlacement(cfg, Cluster())
    stage_num = cfg.rollout.pipeline_stage_num
    env_world_size = component_placement.get_world_size("env")

    if cfg.runner.val_check_interval > 0 or cfg.runner.only_eval:
        assert cfg.env.eval.total_num_envs > 0, (
            "Total number of parallel environments for evaluation must be greater than 0"
        )
        assert cfg.env.eval.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes"
        )
        assert cfg.env.eval.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.eval.total_num_envs // env_world_size // stage_num > 0, (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.eval.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.eval.group_size
            == 0
        ), (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.eval.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks
            == 0
        ), (
            "env.eval.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )

    if not cfg.runner.only_eval:
        assert cfg.env.train.total_num_envs > 0, (
            "Total number of parallel environments for training must be greater than 0"
        )
        assert cfg.env.train.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes"
        )
        assert cfg.env.train.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.train.total_num_envs // env_world_size // stage_num > 0, (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.train.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.train.group_size
            == 0
        ), (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.train.max_steps_per_rollout_epoch
            % cfg.actor.model.num_action_chunks
            == 0
        ), (
            "env.train.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )

    with open_dict(cfg):
        weight_sync_interval = cfg.runner.get("weight_sync_interval", 1)
        assert weight_sync_interval > 0, "weight_sync_interval must be greater than 0"
        cfg.runner.weight_sync_interval = weight_sync_interval
    return cfg


def validate_sft_cfg(cfg: DictConfig) -> DictConfig:
    assert cfg.actor.get("global_batch_size", None) is not None, (
        "the actor.global_batch_size is not set"
    )
    assert cfg.actor.get("micro_batch_size", None) is not None, (
        "the actor.micro_batch_size is not set"
    )

    with open_dict(cfg):
        if cfg.data.get("train_data_paths", None) is None:
            # if train_data_paths is None, the code will just eval the model
            assert cfg.data.get("eval_data_paths", None) is not None, (
                "the data.train_data_paths is None, so data.eval_data_paths is required"
            )
        elif cfg.data.get("eval_data_paths", None) is not None:
            # set the val_check_interval to max_epochs
            if cfg.runner.get("val_check_interval", None) is None:
                cfg.runner.val_check_interval = cfg.runner.max_epochs
        else:
            # set the val_check_interval to -1 if there is no eval data
            cfg.runner.val_check_interval = -1
    return cfg

def validate_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.runner.per_worker_log = cfg.runner.get("per_worker_log", False)
        cfg.runner.per_worker_log_path = None
        if cfg.runner.per_worker_log:
            cfg.runner.per_worker_log_path = os.path.join(
                cfg.runner.logger.log_path, "worker_logs"
            )

    # Init cluster
    Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)

    assert cfg.runner.task_type in SUPPORTED_TASK_TYPE, (
        f"task_type must be one of {SUPPORTED_TASK_TYPE}"
    )
    if cfg.runner.task_type == "embodied":
        cfg = validate_embodied_cfg(cfg)

    if cfg.algorithm.adv_type in ("grpo", "grpo_dynamic", "reinpp_baseline"):
        assert cfg.algorithm.group_size > 1

    assert cfg.actor.training_backend in SUPPORTED_TRAINING_BACKENDS, (
        f"Unsupported training_backend {cfg.actor.training_backend}. Supported training backends are {SUPPORTED_TRAINING_BACKENDS}."
    )

    if cfg.actor.training_backend == "fsdp":
        component_placement = HybridComponentPlacement(cfg, Cluster())
        actor_world_size = component_placement.get_world_size("actor")
        assert (
            cfg.actor.global_batch_size
            % (cfg.actor.micro_batch_size * actor_world_size)
            == 0
        ), (
            f"actor.global_batch_size ({cfg.actor.global_batch_size}) must be divisible by (actor.micro_batch_size ({cfg.actor.micro_batch_size}) * actor_world_size ({actor_world_size}))"
        )
        cfg.actor = validate_fsdp_cfg(cfg.actor)

    if cfg.critic.use_critic_model and cfg.critic.training_backend == "fsdp":
        cfg.critic = validate_fsdp_cfg(cfg.critic)

    return cfg


def build_config(cls, cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        cfg = asdict(cfg)

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in cfg:
            kwargs[f.name] = cfg.get(f.name)

    return cls(**kwargs)
