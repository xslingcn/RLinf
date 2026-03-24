import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.pi05.configuration_pi05 import (
    FeatureType,
    PI05Config,
    PolicyFeature,
)
from rlinf.models.embodiment.pi05.modeling_pi05 import (
    PI05Pytorch,
    _policy_no_init_context,
    _restore_missing_tied_weight_keys,
    resize_with_pad_torch,
)


def _resolve_checkpoint_dir(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    return snapshot_download(repo_id=model_path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_stats(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    tensors = load_file(str(path), device="cpu")
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in tensors.items():
        feature_name, stat_name = key.rsplit(".", 1)
        stats.setdefault(feature_name, {})[stat_name] = value.detach().cpu()
    return stats


def _normalize_model_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        updated_key = key[len("model.") :] if key.startswith("model.") else key
        normalized[updated_key] = value

    lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    embed_key = (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    )
    if lm_head_key in normalized and embed_key not in normalized:
        normalized[embed_key] = normalized[lm_head_key]
    return normalized


def _feature_map_from_config(
    raw_features: dict[str, dict[str, Any]],
) -> dict[str, PolicyFeature]:
    feature_map = {}
    for name, spec in raw_features.items():
        feature_map[name] = PolicyFeature(
            type=FeatureType(spec["type"]),
            shape=tuple(spec["shape"]),
        )
    return feature_map


def _build_config(checkpoint_dir: str) -> PI05Config:
    config_path = Path(checkpoint_dir) / "config.json"
    raw_config = _load_json(config_path)
    config_field_names = {field.name for field in fields(PI05Config)}
    config_kwargs = {k: v for k, v in raw_config.items() if k in config_field_names}
    config_kwargs["input_features"] = _feature_map_from_config(
        raw_config["input_features"]
    )
    config_kwargs["output_features"] = _feature_map_from_config(
        raw_config["output_features"]
    )
    return PI05Config(**config_kwargs)


def _normalize_quantiles(
    tensor: torch.Tensor,
    stats: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    q01 = stats["q01"].to(device=tensor.device, dtype=tensor.dtype)
    q99 = stats["q99"].to(device=tensor.device, dtype=tensor.dtype)
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.full_like(denom, eps), denom)
    return 2.0 * (tensor - q01) / denom - 1.0


def _unnormalize_quantiles(
    tensor: torch.Tensor,
    stats: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    q01 = stats["q01"].to(device=tensor.device, dtype=tensor.dtype)
    q99 = stats["q99"].to(device=tensor.device, dtype=tensor.dtype)
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.full_like(denom, eps), denom)
    return (tensor + 1.0) * denom / 2.0 + q01


def _pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return torch.nn.functional.pad(vector, (0, new_dim - vector.shape[-1]))


class PI05InferenceAdapter(torch.nn.Module, BasePolicy):
    """RLinf inference adapter around the vendored LeRobot PI05 core."""

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        action_chunk: int | None = None,
        num_steps: int | None = None,
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.config = _build_config(checkpoint_dir)
        self.runtime_action_chunk = action_chunk or self.config.n_action_steps
        self.runtime_num_steps = num_steps or self.config.num_inference_steps
        self.preprocessor_stats = _load_stats(
            Path(checkpoint_dir)
            / "policy_preprocessor_step_2_normalizer_processor.safetensors"
        )
        self.postprocessor_stats = _load_stats(
            Path(checkpoint_dir)
            / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.image_feature_order = tuple(
            key
            for key in self.config.input_features
            if key.startswith("observation.images.")
        )

        with _policy_no_init_context():
            self.model = PI05Pytorch(self.config)
        self._load_weights()

    def _load_weights(self) -> None:
        state_dict = load_file(
            str(Path(self.checkpoint_dir) / "model.safetensors"), device="cpu"
        )
        updated_state_dict = _normalize_model_state_dict_keys(state_dict)
        updated_state_dict, _ = _restore_missing_tied_weight_keys(updated_state_dict)
        missing, unexpected = self.model.load_state_dict(
            updated_state_dict, strict=False
        )
        self.model.paligemma_with_expert.paligemma.tie_weights()
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to load PI05 checkpoint cleanly: {len(missing)} missing, {len(unexpected)} unexpected"
            )

    def _obs_to_semantic_images(
        self, env_obs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        wrist_images = env_obs["wrist_images"]
        top = env_obs["main_images"]
        left = wrist_images[:, 0]
        right = wrist_images[:, 1]
        return {
            "observation.images.left": left,
            "observation.images.right": right,
            "observation.images.top": top,
        }

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return _normalize_quantiles(state, self.preprocessor_stats["observation.state"])

    def _tokenize_task(
        self, task_descriptions: list[str], state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_state = self._normalize_state(state)
        padded_state = _pad_vector(normalized_state, self.config.max_state_dim)
        discretized = (
            np.digitize(
                padded_state.detach().cpu().numpy(),
                bins=np.linspace(-1, 1, 256 + 1)[:-1],
            )
            - 1
        )
        prompts = []
        for idx, task in enumerate(task_descriptions):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized[idx]))
            prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")
        tokenized = self.tokenizer(
            prompts,
            max_length=self.config.tokenizer_max_length,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="pt",
        )
        return tokenized["input_ids"], tokenized["attention_mask"].to(dtype=torch.bool)

    def _prepare_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        original_dtype = image.dtype
        image = image.to(dtype=torch.float32, device="cpu")
        if original_dtype == torch.uint8 or image.max() > 1.0:
            image = image / 255.0
        # Match LeRobot PI05Policy._preprocess_images exactly:
        # keep images in [0, 1] through resize_with_pad_torch, then map to [-1, 1].
        image = resize_with_pad_torch(image, *self.config.image_resolution)
        image = image * 2.0 - 1.0
        return image.to(device=next(self.parameters()).device, dtype=torch.float32)

    def _build_model_inputs(
        self, env_obs: dict[str, Any]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        semantic_images = self._obs_to_semantic_images(env_obs)
        images = []
        img_masks = []
        for image_key in self.image_feature_order:
            image = self._prepare_image_tensor(semantic_images[image_key])
            images.append(image)
            img_masks.append(
                torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
            )

        tasks = env_obs["task_descriptions"]
        if isinstance(tasks, str):
            tasks = [tasks]
        tokens, masks = self._tokenize_task(tasks, env_obs["states"])
        device = next(self.parameters()).device
        return images, img_masks, tokens.to(device), masks.to(device)

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        compute_values: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del mode, compute_values
        images, img_masks, tokens, masks = self._build_model_inputs(env_obs)
        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            num_steps=self.runtime_num_steps,
            **kwargs,
        )
        action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, : self.runtime_action_chunk, :action_dim].to(
            dtype=torch.float32
        )
        actions = _unnormalize_quantiles(actions, self.postprocessor_stats["action"])
        return actions.cpu(), {}

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "TODO(agent): PI05 vendored migration currently supports inference only; RL training routing still needs the adapter layer."
        )


def get_model(cfg, torch_dtype=None):
    del torch_dtype
    checkpoint_dir = _resolve_checkpoint_dir(str(cfg.model_path))
    openpi_cfg = getattr(cfg, "openpi", None)
    action_chunk = (
        getattr(openpi_cfg, "action_chunk", None) if openpi_cfg is not None else None
    )
    num_steps = (
        getattr(openpi_cfg, "num_steps", None) if openpi_cfg is not None else None
    )
    model = PI05InferenceAdapter(
        checkpoint_dir,
        action_chunk=action_chunk,
        num_steps=num_steps,
    )
    return model
