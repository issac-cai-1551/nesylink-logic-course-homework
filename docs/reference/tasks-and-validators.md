# NesyLink Tasks

Tasks are registered in Python. They compose map selection, reward selection,
episode defaults, Gymnasium ID, and mission text without putting task logic into
map JSON.

## Built-in Tasks

- `collect_key_easy` -> `NesyLink-CollectKeyEasy-v0`
- `kill_monsters_easy` -> `NesyLink-KillMonstersEasy-v0`
- `avoid_traps_easy` -> `NesyLink-AvoidTrapsEasy-v0`

## Use a Task

```python
from nesylink.env import make_env

env = make_env(task_id="collect_key_easy")
```

or through Gymnasium:

```python
import gymnasium as gym
import nesylink

env = gym.make("NesyLink-CollectKeyEasy-v0")
```

## Register a Custom Task

```python
from nesylink.tasks import TaskSpec, register_task

register_task(TaskSpec(
    task_id="my_task",
    gym_id="NesyLink-MyTask-v0",
    map_id="dungeon",
    reward_id="sparse_exit",
    max_steps=500,
    mission="Reach the exit.",
))
```

Map JSON still only describes the world: layout, spawns, objects, exits, and
room graph data. Reward and task metadata belong in `TaskSpec` or direct
`make_env(...)` arguments.
