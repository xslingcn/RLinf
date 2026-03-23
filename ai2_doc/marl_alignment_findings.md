# `feat/marl` Prompt Alignment Findings

Date: 2026-03-23

## Scope

This note covers prompt-level alignment and TOPReward numeric alignment between:

- the old RLinf `VLMPlannerWorker` path
- the new `marl` planner / TOPReward path

It intentionally stays at the prompt / input-contract level plus observed
TOPReward score parity, and does not cover broader runtime decoding topics.

## Bottom Line

Current state:

- TOPReward prompt semantics are aligned.
- Planner prompt wording is aligned.
- Planner input surface is still not fully aligned.

So the right summary is:

- if the question is “did we align the text prompt templates?”:
  - mostly yes
- if the question is “did we make the whole planner input contract identical?”:
  - no

## TOPReward Prompt Alignment

At the prompt/token-selection level, the new `marl` path matches the old RLinf
TOPReward path.

Relevant code:

- RLinf:
  - `rlinf/algorithms/rewards/top_reward/top_reward.py`
- `marl`:
  - `marl/src/marl/topreward.py`
  - `marl/src/marl/runtime.py`

What is aligned:

- rendered prompt text
- video / frame prompt structure
- final-token log-prob target selection

Controlled parity check result:

- `TOPREWARD_PROMPT_EQUAL True`
- `TOPREWARD_SCORE_RLINF -0.0018493706593289971`
- `TOPREWARD_SCORE_MARL_HELPER -0.0018493706593289971`

So for TOPReward, the prompt-level story is straightforward:

- prompt aligned

## TOPReward Numerical Alignment

Prompt alignment does not imply exact numerical parity.

On the local real-model A/B, the old HF path and the new `marl` / `sglang`
path remained close, but not identical, on TOPReward scores.

Using the prompt-aligned setup, the observed values were:

- `humanoid_walk`
  - HF: `-15.0`
  - `sglang`: `-14.750000953674316`
  - drift: `+0.2499990463256836`
- `cartpole_balance`
  - HF: `-15.5`
  - `sglang`: `-15.750000953674316`
  - drift: `-0.2500009536743164`

So the numeric conclusion is:

- TOPReward prompt semantics are aligned
- TOPReward values are close
- TOPReward values are not exactly identical across backends

## Planner Prompt Alignment

### What is aligned now

The current `marl` planner prompt has been pulled into line with the old RLinf
planner wording.

Relevant code:

- old RLinf planner:
  - `rlinf/workers/vlm_planner/vlm_planner_worker.py`
- current `marl` planner:
  - `marl/src/marl/config.py`
  - `marl/src/marl/runtime.py`

Aligned parts:

- system prompt now uses the old RLinf wording:
  - `AI assistant controlling a bimanual robot arm`
  - explicit short imperative requirement
  - explicit `5-15 words` constraint
- user text now uses the old RLinf wording:
  - `History of past steps: ...`
  - `What is the single best next subtask for the robot to execute?`
- the previous `High-level task:` planner injection has been removed

So at the text-template level:

- planner system prompt aligned
- planner user prompt aligned

### What is still not aligned

Even though the prompt text now matches, the planner is still not seeing the
same effective inputs as the old worker-based path.

#### 1. Image inputs are still richer in `marl`

Old worker-based replanning effectively only used:

- `main_images`

Current `marl` planner can still receive:

- `main`
- `wrist_*`
- `extra_*`

That means the prompt text may match, but the visual context does not.

#### 2. Memory behavior is still richer in `marl`

Old RLinf path:

- had a planner memory API
- but in practice the old pipeline did not append meaningful entries during
  replanning

Current `marl` path:

- actively passes `memory_text`
- accumulates entries like:
  - `Step N: task=...`
  - `Planner updated task to: ...`

So again, the prompt shell matches, but the actual text content reaching the
planner is still richer than before.

## Final Conclusion

If the question is specifically about prompt alignment:

- TOPReward prompt: aligned
- planner system prompt: aligned
- planner user prompt: aligned

If the question is about TOPReward numeric parity:

- close, but not identical

If the question is about full planner-input alignment:

- not aligned yet

The remaining non-alignment is not mainly in the prompt template anymore. It is
in the runtime context:

- richer memory
- richer image inputs
