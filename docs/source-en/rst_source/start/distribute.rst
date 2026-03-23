Multi-node Training
===================

This guide shows how to launch a **4-node Ray cluster** (each node
has **8 GPUs**) and run distributed **embodied RL** training.  
The same procedure scales to any number of nodes/GPUs, as long as you customize the YAML configuration according to your needs.


Prerequisites
-------------

Before running, make sure to check the following:

* Clone RLinf to a shared filesystem accessible by all nodes.
* Ensure that each node has started the corresponding container image.



Step 1: Start a Ray Cluster
----------------------------

Clean up *old* cached state first:

.. code-block:: bash

   rm -f ray_utils/ray_head_ip.txt

Open a shell on *each* node and run:

==========================================  ==========================
node index                                  command
==========================================  ==========================
0 (head)                                    ``RANK=0 bash ray_utils/start_ray.sh``
1                                           ``RANK=1 bash ray_utils/start_ray.sh``
2                                           ``RANK=2 bash ray_utils/start_ray.sh``
3                                           ``RANK=3 bash ray_utils/start_ray.sh``
==========================================  ==========================


Once the scripts run successfully, the terminal on the **head node** should display output similar to the following (for simplicity, we only show the example of 2 nodes with 16 GPUs):

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/start-0.jpg" width="800"/>

On each **worker node**, the terminal should display:

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/start-1.jpg" width="800"/>

After all four startup scripts print *Ray started*, **remain** in the head node terminal and verify the total cluster size (in this example, ``4 × 8 = 32`` GPUs):

.. code-block:: bash

   bash ray_utils/check_ray.sh 32

.. note::

   The argument to ``check_ray.sh`` must equal the number of accelerators/GPUs in the cluster. 

If successful, your terminal should show:

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/check.jpg" width="800"/>

Note: For simplicity, the images in this example only show a 2-node setup with 16 GPUs.


Step 2: Launch Training Tasks
------------------------------------

Here we provide startup examples in two modes: collocated mode and disaggregated mode.

Collocated 
^^^^^^^^^^^^^^

Every training stage (actor, env, rollout) shares **all GPUs**.
Edit the sample YAML:

.. code-block:: yaml

   # examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml
   cluster:
     num_nodes: 4          # adapt to your cluster
     component_placement:
       actor,env,rollout: all  # “all” means the whole visible GPU set

Launch from the head node:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
        maniskill_ppo_openvla_quickstart


Disaggregated
^^^^^^^^^^^^^^^^^^

Different stages receive disjoint GPU ranges,
allowing fine-grained pipelining. Edit the pipeline YAML:

.. code-block:: yaml

   # examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml
   cluster:
     num_nodes: 4
     component_placement:
       env:        0-7         # 8  GPUs
       rollout:    8-23        # 16 GPUs
       actor:      24-31       # 8  GPUs

* ``env + rollout + actor`` **must equal** the total GPU count
  (here ``32``).
* Ranges are inclusive.

Start the job:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
        maniskill_ppo_openvla_quickstart
