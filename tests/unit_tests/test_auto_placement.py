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

import os
import sys
from unittest.mock import MagicMock

import pytest

# Add auto_placement tools to path for testing
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../toolkits/auto_placement")
)
from auto_placement_worker import AutoPlacementWorker, get_workflow_graph
from node import ComponentNode, MegatronNode, SccNode
from placement import ScheduleMode, ScheduleResult
from util import init_global_config
from workflow import Workflow, traverse_st_cuts


def get_mock_config_embodiment():
    mock_cfg = MagicMock()
    mock_cfg.runner.task_type = "embodied"

    mock_cfg.data.rollout_batch_size = 1024
    mock_cfg.data.env_num = 64
    mock_cfg.profile_data.env_profile_data = {
        4: 0.61,
        8: 1.23,
        16: 2.46,
        32: 4.66,
        64: 18.5,
    }
    mock_cfg.profile_data.rollout_profile_data = {
        4: 0.6,
        8: 1.01,
        16: 2.12,
        32: 3.72,
        64: 15.3,
    }

    # Model size
    mock_component_placement = MagicMock()
    mock_component_placement._components = ["env", "rollout", "actor"]
    mock_component_placement.get_world_size.side_effect = lambda component: {
        "env": 4,
        "rollout": 4,
        "actor": 4,
    }[component]
    mock_cfg.algorithm.group_size = 1
    mock_cfg.profile_data.actor_cost = 100

    # cluster
    mock_cluster = MagicMock()
    mock_cluster.num_accelerators = 4

    return mock_cfg, mock_component_placement, mock_cluster


class TestNode:
    """Tests for node class."""

    @staticmethod
    def _init_mock_global_config():
        mock_cfg, mock_component_placement, mock_cluster = get_mock_config_embodiment()
        init_global_config(mock_cfg, mock_component_placement, mock_cluster)

    def test_node_creation(self):
        """Test basic node creation and methods."""
        self._init_mock_global_config()
        actor_node = MegatronNode("actor")

        assert actor_node.role == "actor"

    def test_node_validation(self):
        """Test node validation."""
        self._init_mock_global_config()
        valid_gpu_nums = [1, 2, 4, 8]
        actor_node = MegatronNode(role="actor", valid_gpu_nums=valid_gpu_nums)

        for gpu_num in range(10):
            if gpu_num in valid_gpu_nums:
                assert actor_node._validate_gpu_num(gpu_num)
            else:
                assert not actor_node._validate_gpu_num(gpu_num)


class TestWorkflow:
    """Tests for the Workflow class."""

    class _TestNode:
        def __init__(self, role: str):
            self.role = role

        def profile(self, gpu_num: int):
            return gpu_num

        def __str__(self):
            return self.role

        def __repr__(self):
            return self.__str__()

        def __hash__(self):
            return hash(self.role)

        def __eq__(self, other):
            return isinstance(other, TestWorkflow._TestNode) and self.role == other.role

    _name_to_node_dict = {
        "env": _TestNode("env"),
        "env_rollout": _TestNode("env_rollout"),
        "actor": _TestNode("actor"),
    }

    def get_node(self, name: str) -> ComponentNode:
        return self._name_to_node_dict[name]

    def test_workflow_graph(self):
        """Test workflow creation and basic properties."""
        cfg = MagicMock()
        workflow_graph = get_workflow_graph(cfg)
        assert workflow_graph == {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }

    def test_workflow_creation(self):
        """Test workflow creation."""
        graph = {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }

        workflow_graph = {}
        for node, neighbors in graph.items():
            workflow_graph[self.get_node(node)] = [
                self.get_node(neighbor) for neighbor in neighbors
            ]
        workflow = Workflow(workflow_graph)
        assert set(workflow.nodes) == {
            self.get_node("env"),
            self.get_node("env_rollout"),
            self.get_node("actor"),
        }
        assert workflow.topological_order == [
            self.get_node("env"),
            self.get_node("env_rollout"),
            self.get_node("actor"),
        ]

    def test_traverse_st_cuts(self):
        """Test traverse st cuts of workflow."""
        graph = {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }
        workflow = Workflow(graph)
        cuts = traverse_st_cuts(workflow)
        assert len(cuts) == 2
        assert cuts[0][0].is_node() and cuts[0][0].nodes[0] == "env"
        assert cuts[1][1].is_node() and cuts[1][1].nodes[0] == "actor"

        cuts = traverse_st_cuts(cuts[0][1])
        assert len(cuts) == 1
        assert cuts[0][0].is_node() and cuts[0][0].nodes[0] == "env_rollout"
        assert cuts[0][1].is_node() and cuts[0][1].nodes[0] == "actor"

    def test_compress_sccs(self):
        """Test SCC compression."""
        graph = {
            self.get_node("env_rollout"): [self.get_node("env")],
            self.get_node("env"): [
                self.get_node("env_rollout"),
                self.get_node("actor"),
            ],
            self.get_node("actor"): [],
        }
        workflow = Workflow(graph)
        compressed_workflow = workflow.compress_sccs()

        assert len(workflow.nodes) == 3 and len(compressed_workflow.nodes) == 2

        topological_order = compressed_workflow.topological_order
        assert isinstance(topological_order[0], SccNode)
        assert topological_order[0].role in ["env - env_rollout", "env_rollout - env"]


class TestAutoPlacementWorkerForEmbodiment:
    """Tests for the SchedulerTask class."""

    def test_embodiment(self):
        """Test SchedulerTask initialization."""
        # Create a mock config
        mock_cfg, mock_component_placement, mock_cluster = get_mock_config_embodiment()

        init_global_config(mock_cfg, mock_component_placement, mock_cluster)

        graph = {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }

        auto_placement_worker = AutoPlacementWorker(
            mock_cfg, mock_component_placement, graph
        )
        res = auto_placement_worker.run()
        assert res.total_gpu_num == mock_cluster.num_accelerators
        assert isinstance(res, ScheduleResult)
        assert res.mode == ScheduleMode.COLLOCATED


if __name__ == "__main__":
    pytest.main(["-v", __file__])
