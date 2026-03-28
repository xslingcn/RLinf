# Copyright 2026 Shirui Chen
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

"""Regression tests for remote disconnect handling in AsyncEnvWorker."""

import asyncio

import pytest

from rlinf.envs.remote.remote_env import RobotServerDisconnectedError
from rlinf.workers.env.async_env_worker import AsyncEnvWorker


def test_async_env_worker_waits_for_staged_shutdown_on_remote_disconnect() -> None:
    worker = object.__new__(AsyncEnvWorker)
    log_messages: list[str] = []

    async def fake_run_interact_once(*args, **kwargs):
        del args, kwargs
        raise RobotServerDisconnectedError(
            "[RemoteEnv] Robot server disconnected during ChunkStep (gRPC UNAVAILABLE)."
        )

    worker._run_interact_once = fake_run_interact_once
    worker.log_warning = log_messages.append

    async def _exercise() -> None:
        task = asyncio.create_task(
            AsyncEnvWorker._interact(worker, None, None, None, None)
        )
        await asyncio.sleep(0.05)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_exercise())

    assert len(log_messages) == 1
    assert "Waiting for the staged entrypoint to stop training." in log_messages[0]
