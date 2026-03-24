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

from enum import Enum


class SupportedEnvType(Enum):
    YAM = "yam"
    REMOTE = "remote"


def get_env_cls(env_type: str, env_cfg=None):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment.
        env_cfg: Optional environment configuration.

    Returns:
        Environment class corresponding to the environment type.
    """

    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.YAM:
        from rlinf.envs.yam.yam_env import YAMEnv

        return YAMEnv
    if env_type == SupportedEnvType.REMOTE:
        from rlinf.envs.remote.remote_env import RemoteEnv

        return RemoteEnv

    raise NotImplementedError(f"Environment type {env_type} not implemented")
