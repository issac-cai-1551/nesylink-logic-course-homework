# Map Creation Guide

NesyLink maps are JSON world definitions. They describe the dungeon layout and
objects only; reward logic and training objectives belong in Python rewards or
task specs.

## File Layout

Built-in maps live under:

```text
nesylink/map_data/dungeons/
```

Supported lookup patterns:

- `nesylink/map_data/dungeons/<map_id>/dungeon.json`
- `nesylink/map_data/dungeons/<map_id>/room_001.json`
- `nesylink/map_data/dungeons/<map_id>.json`

That means this works when `room_001.json` exists under `key_door/`:

```python
env = make_env(map_id="key_door", reward_id="collect_key")
```

## Standalone Room

A minimal single-room map:

```json
{
  "id": "room_001",
  "coord": [0, 0],
  "layout": [
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    ".........."
  ],
  "spawns": {
    "default": [4, 6]
  },
  "default_spawn": "default",
  "objects": [],
  "exits": []
}
```

The playable area is fixed at 10 columns by 8 rows. `.` is floor and `#` is
wall.

## Objects

Supported object kinds:

- `chest`
- `monster`
- `trap`
- `button`
- `npc`

Examples:

```json
{
  "id": "chest_key",
  "kind": "chest",
  "pos": [1, 3],
  "loot": {"kind": "key", "key_id": "task_key"}
}
```

```json
{
  "id": "monster_1",
  "kind": "monster",
  "pos": [7, 4],
  "monster_type": "chaser",
  "hp": 2,
  "damage": 1
}
```

## Exits

Exit directions are fixed to `north`, `south`, `west`, and `east`. The engine
uses fixed two-tile doorway shapes for each direction.

Normal exit:

```json
{
  "id": "north_exit",
  "direction": "north",
  "target_room": "room_001",
  "target_entry": "from_south",
  "type": "normal",
  "success_message": "CLEARED!"
}
```

Locked key exit:

```json
{
  "id": "east_exit",
  "direction": "east",
  "target_room": "room_2",
  "target_entry": "from_west",
  "type": "locked_key",
  "requires": {"key_count": 1, "consume_key": true},
  "blocked_message": "NEED KEY"
}
```

Conditional exit:

```json
{
  "id": "south_exit",
  "direction": "south",
  "target_room": "room_3",
  "target_entry": "from_north",
  "type": "conditional",
  "requires": {"button_pressed": "button_1"}
}
```

Set `complete_task: true` on an exit when reaching it should produce an
environment-completion event.

## Multi-room Dungeon

A dungeon root file references room files:

```json
{
  "schema_version": 1,
  "dungeon_id": "prototype",
  "start_room": "room_0_0",
  "room_files": [
    "rooms/room_0_0.json",
    "rooms/room_1_0.json"
  ]
}
```

Each referenced room is a normal room JSON file with an `id`, `coord`,
`layout`, `spawns`, `objects`, and `exits`.
