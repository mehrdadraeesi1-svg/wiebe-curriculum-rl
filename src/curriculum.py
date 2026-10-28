"""
Curriculum scheduling utilities.

The environment advances its curriculum stage in ``reset()`` by comparing its
``global_step`` with ``schedule_steps``. Each sub-environment of a vectorised
environment keeps its own counter, so with ``n_envs`` parallel environments a
threshold of 100 000 steps is reached only after ``n_envs * 100 000`` total
steps. ``CurriculumSyncCallback`` removes that dependence by pushing the
learner's total step count (``model.num_timesteps``) to every
sub-environment.

Known issue in v0.x (fixed in v1.0.0)
-------------------------------------
The training scripts of v0.x did not synchronise the counter. With the
configuration used in the associated study (n_envs = 4, 300 000 total
steps, thresholds [0, 100 000, 200 000]) each sub-environment reached only
75 000 steps, so training remained in stage 1 (alpha = 0.50, beta = 0.30)
throughout. See CHANGELOG.md.
"""

from __future__ import annotations

from typing import Dict, List

from stable_baselines3.common.callbacks import BaseCallback

DEFAULT_SCHEDULE: Dict[str, List[float]] = {
    "stages": [0, 1, 2],
    "schedule_steps": [0, 100_000, 200_000],
    "spans": [0.50, 0.75, 1.00],          # alpha_k
    "noise_scales": [0.30, 0.65, 1.00],   # beta_k
}

#: Effective configuration of the v0.x runs reported in Ref. [1]
LEGACY_EFFECTIVE_SCHEDULE: Dict[str, List[float]] = {
    "stages": [0],
    "schedule_steps": [0],
    "spans": [0.50],
    "noise_scales": [0.30],
}


class CurriculumSyncCallback(BaseCallback):
    """Synchronise the curriculum counter of all sub-environments.

    The update is done at the start of every rollout, so stage changes take
    effect with a delay of at most ``n_steps * n_envs`` environment steps.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.stage_history: List[tuple] = []

    def _sync(self) -> None:
        self.training_env.env_method("set_global_step", int(self.num_timesteps))
        stages = self.training_env.get_attr("curriculum_stage")
        stage = int(max(stages))
        if not self.stage_history or self.stage_history[-1][1] != stage:
            self.stage_history.append((int(self.num_timesteps), stage))
            if self.verbose:
                print(f"[curriculum] step {self.num_timesteps}: stage {stage + 1}")
        self.logger.record("curriculum/stage", stage + 1)

    def _on_training_start(self) -> None:
        self._sync()

    def _on_rollout_start(self) -> None:
        self._sync()

    def _on_step(self) -> bool:
        return True
