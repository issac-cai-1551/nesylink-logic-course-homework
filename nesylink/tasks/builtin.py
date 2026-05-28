from __future__ import annotations

from .registry import register_task
from .specs import TaskSpec


BUILTIN_TASKS = (
    TaskSpec(
        task_id="collect_key_easy",
        gym_id="NesyLink-CollectKeyEasy-v0",
        map_id="key_door",
        reward_id="collect_key",
        max_steps=500,
        mission="Collect the key and reach the exit.",
    ),
    TaskSpec(
        task_id="kill_monsters_easy",
        gym_id="NesyLink-KillMonstersEasy-v0",
        map_id="kill_monsters",
        reward_id="kill_monster",
        max_steps=500,
        mission="Defeat the monster, collect the key, and reach the exit.",
    ),
    TaskSpec(
        task_id="avoid_traps_easy",
        gym_id="NesyLink-AvoidTrapsEasy-v0",
        map_id="avoid_traps",
        reward_id="sparse_exit",
        max_steps=500,
        mission="Reach the exit while avoiding traps.",
    ),
)


def register_builtin_tasks() -> None:
    for task in BUILTIN_TASKS:
        try:
            register_task(task)
        except ValueError as exc:
            if "duplicate" not in str(exc):
                raise
