from __future__ import annotations

from .builtin import BUILTIN_TASKS, register_builtin_tasks
from .registry import get_task, get_task_by_gym_id, list_tasks, register_task
from .specs import TaskSpec


register_builtin_tasks()


__all__ = [
    "BUILTIN_TASKS",
    "TaskSpec",
    "get_task",
    "get_task_by_gym_id",
    "list_tasks",
    "register_builtin_tasks",
    "register_task",
]
