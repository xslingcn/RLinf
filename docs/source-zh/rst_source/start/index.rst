快速开始
==========

欢迎使用 RLinf 快速上手指南。本节将带你一步步运行 RLinf，帮助你快速了解整个框架的使用流程。

我们提供了三个简洁示例，展示 RLinf 的基本工作流程，帮助你快速开始使用：

- **安装方式：** RLinf 支持两种安装方法：使用 Docker 镜像，或自定义用户环境（详见 :doc:`installation`）。

- **具身智能训练：** 在 ManiSkill3 环境中，使用 PPO 算法对 OpenVLA 和 OpenVLA-OFT 模型进行训练（详见 :doc:`vla`）。

- **分布式训练：** 支持多节点具身智能任务训练（详见 :doc:`distribute`）。

- **模型评估：** 评估模型在具身智能场景任务下的表现（详见 :doc:`vla-eval`）。


SOTA 强化学习复现
==========================

RLinf 提供了一整套**可复现的 SOTA 强化学习配置**，用户无需额外工程改造，只需直接运行官方脚本和配置文件，即可复现论文级或业界领先的训练效果。

在具身智能任务上，RLinf 在 **LIBERO**、**ManiSkill**、**RoboTwin** 等多个基准中达到了或接近当前最优的成功率，支持 OpenVLA、OpenVLA-OFT、π₀/π₀.₅、GR00T 等多种 VLA 模型（详见 :doc:`../examples/embodied/index` 中的示例库与 :doc:`../tutorials/rlalg/index` 中的算法教程）。

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   vla
   distribute
   vla-eval
