# Copyright 2026 Ying-Chun Lee
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

"""Unit tests for the YAM follower client compatibility layer."""

from unittest.mock import Mock

import portal

from rlinf.envs.yam.yam_follower_client import YAMFollowerClient


def test_follower_client_exposes_gain_management_methods(monkeypatch) -> None:
    monkeypatch.setattr(portal, "Client", lambda *_args, **_kwargs: Mock())

    client = YAMFollowerClient(port=1234)

    assert "get_robot_info" in client.supported_remote_methods
    assert "update_kp_kd" in client.supported_remote_methods
