# NesyLink API

## Environment Construction

Preferred entrypoint:

```python
from nesylink.env import make_env
```

Supported forms:

```python
env = make_env(task_id="collect_key_easy")
env = make_env(map_id="dungeon", reward_id="sparse_exit", max_steps=500)
env = make_env(map_path="nesylink/map_data/dungeons/prototype/dungeon.json", reward_id="collect_key")
env = make_env(map_id="dungeon", reward_module="nesylink.rewards.exploration")
```

Gymnasium registered form:

```python
import gymnasium as gym
import nesylink

env = gym.make("NesyLink-CollectKeyEasy-v0")
```

Parameters:

- `task_id`
- `map_id`
- `map_path`
- `reward_id`
- `reward_module`
- `reward_kwargs`
- `max_steps`
- `render_mode`
- `action_repeat`
- `api`

Explicit parameters take precedence over task defaults. `map_path` takes
precedence over `map_id`.

## reset / step

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

`reset()` synchronizes the reward object by calling `reward_fn.reset(obs, info)`.

`step()`:

- advances game mechanics
- computes reward from the configured reward object
- merges base termination with reward-driven termination
- truncates on `max_steps`
- stores reward metadata in `info["reward"]`

Action semantics:

- `0`: wait
- `1..4`: move up/down/left/right
- `5`: trigger slot `A`
- `6`: trigger slot `B`

Default slot behavior:

- `A` starts with `sword`
- `B` starts with `shield`
- `A` first tries chest/NPC interaction; if nothing is interactable, it uses the equipped `A` item
- `shield` blocks contact damage and never damages monsters
- `sword` handles melee damage with a one-tile forward hitbox
- action poses remain visible for multiple ticks in RGB renders, but damage/block resolution still happens on the triggering step only

## Info Shape

Top-level `info` keys:

- `episode`
- `env`
- `agent`
- `inventory`
- `entities`
- `events`
- `game`
- `terminal_reason`
- `control`
- `debug`
- `reward`

`info["task"]` is deprecated and no longer part of the contract.

Additional fields exposed by this version:

- `info["agent"]["facing"]`
- `info["inventory"]["equipped"]`
- `info["debug"]["action_item"]`
- `info["debug"]["action_pose"]`
- `info["debug"]["action_ticks_remaining"]`

## Observation Shape

The default observation is a `gymnasium.spaces.Dict`.

Common keys:

- `grid`: `uint8`, shape `(8, 10)`
- `player_position_px`: `float32`, shape `(2,)`
- `player_tile`: `int32`, shape `(2,)`
- `health`: `int32`, shape `(1,)`
- `gold`: `int32`, shape `(1,)`
- `keys`: `int32`, shape `(1,)`
- `inventory_ids`: `int32`, shape `(2,)`
- `monsters_position_px`: `float32`, shape `(max_monsters, 2)`
- `monsters_tile`: `int32`, shape `(max_monsters, 2)`
- `monsters_active_mask`: `uint8`, shape `(max_monsters,)`
- `monsters_hp`: `int32`, shape `(max_monsters,)`

Use `env.observation_space` for exact bounds in code.

## Rendering

```python
env = make_env(task_id="collect_key_easy", render_mode="rgb_array")
obs, info = env.reset(seed=0)
frame = env.render()
```

The RGB frame includes the dungeon area and HUD. The structured `grid`
observation covers only the playable 10 by 8 tile area.
