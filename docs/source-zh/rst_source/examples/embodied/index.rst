具身智能场景
============

具身智能场景包含 SOTA 模型（如 OpenVLA、OpenVLA-OFT、GR00T、Dexbotic）和不同模拟器（如 LIBERO、ManiSkill）的训练示例，以及真机强化学习训练示例等。

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
         <source src="https://github.com/RLinf/misc/raw/main/pic/embody.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
       <a href="maniskill.html" style="text-decoration: underline; color: blue;">
          <b>基于ManiSkill的强化学习</b>
         </a><br>
         ManiSkill+OpenVLA+PPO/GRPO达到SOTA训练效果
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/libero_numbers.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="libero.html" style="text-decoration: underline; color: blue;">
          <b>基于LIBERO的强化学习</b>
         </a><br>
         LIBERO+OpenVLA-OFT+GRPO成功率达99%
       </p>
     </div>

   </div>

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/IsaacLab.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="isaaclab.html" style="text-decoration: underline; color: blue;">
          <b>基于IsaacLab的强化学习</b>
         </a><br>
         支持IsaacLab+gr00t+PPO训练
       </p>
     </div>
   </div>

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gr00t.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="gr00t.html" style="text-decoration: underline; color: blue;">
          <b>GR00T-N1.5模型强化学习训练</b>
         </a><br>
         支持GR00T-N1.5强化学习微调
       </p>
     </div>

   </div>

    <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
        <img src="https://github.com/RLinf/misc/raw/main/pic/sac-flow-overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
        <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
          <a href="sac_flow.html" style="text-decoration: underline; color: blue;">
            <b>SAC-Flow 策略训练</b>
          </a><br>
          使用 SAC 训练 Flow Matching 策略 (Sim & Real)
        </p>
      </div>

     
      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
        <img src="https://github.com/RLinf/misc/raw/main/pic/3_layer_mlp.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
        <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
          <a href="mlp.html" style="text-decoration: underline; color: blue;">
            <b>基于MLP的强化学习</b>
          </a><br>
          使用 PPO/SAC/GRPO 训练 PPO 策略
        </p>
      </div>

      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/hpcaitech/Open-Sora-Demo/raw/main/readme/icon.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="opensora.html" style="text-decoration: underline; color: blue;">
          <b>基于 OpenSora 世界模型的强化学习</b>
         </a><br>
         支持 OpenSora 世界模型 + OpenVLA-OFT + GRPO 训练
       </p>
      </div>
    </div>

    <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
      <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/dexmal/dexbotic/main/resources/intro.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
        <a href="dexbotic.html" style="text-decoration: underline; color: blue;">
          <b>基于 Dexbotic 模型的强化学习训练</b>
         </a><br>
         Dexbotic（基于 π₀.₅）+ LIBERO + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/wan.png"
           style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
           data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="wan.html" style="text-decoration: underline; color: blue;">
          <b>基于 Wan 世界模型的强化学习</b>
         </a><br>
         支持 Wan 世界模型 + OpenVLA-OFT + GRPO 训练
       </p>
     </div>
    </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   maniskill
   libero
   isaaclab
   opensora
   wan
   gr00t
   sac_flow
   mlp
   dexbotic
   sft_vlm
