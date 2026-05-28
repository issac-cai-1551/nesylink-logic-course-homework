from __future__ import annotations

from pathlib import Path
from typing import Any

from gymnasium.envs.registration import EnvSpec, register, registry

from .rewards.loader import load_reward
from .tasks import get_task, list_tasks
from .wrappers import DungeonEnv, GymDungeonEnv, get_wrapper
from .core.world.loader import load_map


def make_env(
    config_path: str | Path | None = None,
    *,
    task_id: str | None = None,
    map_id: str | None = None,
    map_path: str | Path | None = None,
    api: str = "gym",
    reward_id: str | None = None,
    reward_module: str | None = None,
    reward_kwargs: dict[str, float] | None = None,
    max_steps: int | None = None,
    action_repeat: int | None = None,
    mission: str | None = None,
    **kwargs: Any,
):
    task = get_task(task_id) if task_id is not None else None
    resolved_map_id = map_id if map_id is not None else task.map_id if task is not None else None
    resolved_map_path = map_path if map_path is not None else task.map_path if task is not None else None
    resolved_reward_id = (
        reward_id if reward_id is not None else task.reward_id if task is not None else None
    )
    resolved_reward_module = (
        reward_module
        if reward_module is not None
        else task.reward_module
        if task is not None
        else None
    )
    resolved_reward_kwargs = (
        reward_kwargs
        if reward_kwargs is not None
        else dict(task.reward_kwargs)
        if task is not None
        else kwargs.pop("reward_kwargs", None)
    )
    resolved_max_steps = max_steps if max_steps is not None else task.max_steps if task is not None else None
    resolved_action_repeat = (
        action_repeat
        if action_repeat is not None
        else task.action_repeat
        if task is not None
        else kwargs.pop("action_repeat", 1)
    )
    resolved_mission = mission if mission is not None else task.mission if task is not None else ""

    resolved_map_path = load_map(
        map_id=resolved_map_id,
        map_path=resolved_map_path if resolved_map_path is not None else config_path,
    )
    reward_fn = load_reward(
        reward_id=resolved_reward_id,
        reward_module=resolved_reward_module,
        reward_kwargs=resolved_reward_kwargs,
    )
    wrapper_cls = get_wrapper(api)
    env = wrapper_cls(
        resolved_map_path,
        reward_fn=reward_fn,
        max_steps=resolved_max_steps,
        action_repeat=resolved_action_repeat,
        mission=resolved_mission,
        map_id=resolved_map_id,
        **kwargs,
    )
    if task is not None:
        env.spec = EnvSpec(
            id=task.gym_id,
            entry_point="nesylink.env:make_env",
            max_episode_steps=resolved_max_steps,
            kwargs={"task_id": task.task_id},
        )
    return env


def register_gym_envs() -> None:
    for task in list_tasks():
        if task.gym_id in registry:
            continue
        register(
            id=task.gym_id,
            entry_point="nesylink.env:make_env",
            kwargs={"task_id": task.task_id},
            max_episode_steps=task.max_steps,
        )


__all__ = ["DungeonEnv", "GymDungeonEnv", "make_env", "register_gym_envs"]
