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

import concurrent.futures
import json
import os
import shutil

import torch
from safetensors.torch import save_file

def get_model_save_helper(model_type: str):
    del model_type
    return None


def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def save_state_dict_sharded_safetensors(
    state_dict: dict,
    out_dir: str,
    base_name: str = "model",
    max_shard_size: float | int = 4 * 1024**3,
) -> tuple[int, int]:
    """
    Save the state dict in sharded safetensors format. It will
    first record every tensor that needs to be stored, and create shard plan.
    After this, it will use thread pool to write to safetensors according
    to the sharded plan.

    Args:
        state_dict(dict[str,torch.tensor]): The state dict to save.
        out_dir(str): where to save the sharded safetensors files.
        base_name(str): The base name for the sharded files.
        max_shard_size(int|float): The maximum size of each shard in bytes. Default is 4GB.

    Returns:
        tuple[int,int]: number of shards created and total size in bytes.
    """
    os.makedirs(out_dir, exist_ok=True)

    items = [(k, v) for k, v in state_dict.items() if torch.is_tensor(v)]
    items.sort(key=lambda kv: kv[0])

    # Plan shards
    shards_plan = []
    current_shard_keys = []
    current_shard_bytes = 0

    def flush_plan():
        nonlocal current_shard_keys, current_shard_bytes
        if not current_shard_keys:
            return
        shards_plan.append((current_shard_keys, current_shard_bytes))
        current_shard_keys = []
        current_shard_bytes = 0

    for name, t in items:
        # Calculate size without moving to CPU
        nbytes = _tensor_nbytes(t)

        if nbytes > max_shard_size:
            flush_plan()
            current_shard_keys.append(name)
            current_shard_bytes = nbytes
            flush_plan()
            continue

        if current_shard_bytes + nbytes > max_shard_size and current_shard_keys:
            flush_plan()

        current_shard_keys.append(name)
        current_shard_bytes += nbytes

    flush_plan()

    num_shards = len(shards_plan)
    total_size = sum(b for _, b in shards_plan)
    weight_map = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for idx, (keys, _) in enumerate(shards_plan):
            shard_idx = idx + 1
            shard_dict = {}

            # (CPU transfer happens here)
            for k in keys:
                t = state_dict[k]
                t = t.detach()
                if t.device.type != "cpu":
                    t = t.cpu()
                if not t.is_contiguous():
                    t = t.contiguous()
                shard_dict[k] = t

            fname = f"{base_name}-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
            fpath = os.path.join(out_dir, fname)

            future = executor.submit(
                save_file, shard_dict, fpath, metadata={"format": "pt"}
            )
            futures.append(future)

            for k in keys:
                weight_map[k] = fname

        # Wait for all tasks to complete and check for exceptions
        for future in concurrent.futures.as_completed(futures):
            future.result()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(
        os.path.join(out_dir, f"{base_name}.safetensors.index.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return num_shards, total_size


def copy_model_config_and_code(
    model_path: str,
    save_path: str,
    suffixes: tuple[str, ...] = (
        ".py",
        ".json",
        ".md",
    ),
) -> None:
    """
    Recursively copies files with specific suffixes from model_path to save_path.
    """
    if not os.path.exists(model_path):
        return

    os.makedirs(save_path, exist_ok=True)

    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith(suffixes):
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(src_file, model_path)
                dst_file = os.path.join(save_path, rel_path)

                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)
