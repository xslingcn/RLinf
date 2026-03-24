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

from omegaconf import DictConfig

from rlinf.config import SupportedModel, get_supported_model, torch_dtype_from_precision
from rlinf.scheduler import Worker


def get_model(cfg: DictConfig):
    model_type = get_supported_model(cfg.model_type)
    if model_type == SupportedModel.LEROBOT_PI05:
        from rlinf.models.embodiment.lerobot_pi05 import get_model
    else:
        return None

    torch_dtype = torch_dtype_from_precision(cfg.precision)
    model = get_model(cfg, torch_dtype)

    if Worker.torch_platform.is_available():
        model = model.to(Worker.torch_device_type)

    if cfg.is_lora:
        raise NotImplementedError(
            "LoRA is not supported for lerobot_pi05 in inference-only mode."
        )

    return model
