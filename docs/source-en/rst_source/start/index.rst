Quickstart
==========

Welcome to the RLinf Quickstart Guide. This section will walk you through launching RLinf for the first time. 
We present three concise examples to demonstrate the framework's workflow and help you get started quickly.


- **Installation:** Two installation methods for RLinf are supported: using a Docker image or a custom user environment (see :doc:`installation`).

- **Embodied training:** Training in the ManiSkill3 environment with the OpenVLA and OpenVLA-OFT models using the PPO algorithm (see :doc:`vla`).

- **Distributed training:** Multi-node training for embodied tasks (see :doc:`distribute`).

- **Evaluation:** Assessing model performance on embodied intelligence (see :doc:`vla-eval`).


SOTA RL Training Reproduction
=====================================

RLinf provides end-to-end recipes that reproduce or match **state-of-the-art (SOTA) RL results** out of the box—users can directly run our configs and scripts to obtain published numbers without custom engineering.

For embodied tasks, RLinf reaches or matches SOTA success rates on benchmarks such as **LIBERO**, **ManiSkill**, **RoboTwin**, and more with OpenVLA, OpenVLA-OFT, π₀/π₀.₅, GR00T and other VLAs (see the :doc:`../examples/embodied/index` gallery and :doc:`../tutorials/rlalg/index` for algorithm details).

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   vla
   distribute
   vla-eval
