from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Deque, Dict, List, Tuple, Set
import numpy as np
from pathlib import Path
from PIL import Image
from vision_exact import StaticTileClassifier, PlayerDetector, MonsterDetector, ExitDetector

# 动作编号
ACTION_NOOP = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_A = 5
ACTION_B = 6

TILE_SIZE = 16
ROOM_W = 10
ROOM_H = 8

# 符号 tile 编码，先和文档里的 grid code 保持一致
EMPTY = 0
WALL = 1
PLAYER = 2
MONSTER = 3
CHEST = 4
EXIT = 5
TRAP = 6
BUTTON = 7
NPC = 8
GAP = 9
BRIDGE = 10
SWITCH = 11




Pos = Tuple[int, int]


@dataclass
class SymbolicObs:
    grid: np.ndarray                    # shape: (8, 10)
    player: Optional[Pos] = None
    facing: str = "up"
    monsters: List[Pos] = field(default_factory=list)
    chests: List[Pos] = field(default_factory=list)
    exits: List[Pos] = field(default_factory=list)
    traps: List[Pos] = field(default_factory=list)
    buttons: List[Pos] = field(default_factory=list)
    switches: List[Pos] = field(default_factory=list)


@dataclass
class BeliefState:
    task_id: Optional[str] = None
    step: int = 0

    # 当前房间记忆
    last_player: Optional[Pos] = None
    facing: str = "up"

    # 任务进度记忆
    has_key: bool = False
    has_sword: bool = False

    keys: int = 0
    gold: int = 0
    items: Set[str] = field(default_factory=set)
    tools: Set[str] = field(default_factory=set)
    
    opened_chests: Set[Pos] = field(default_factory=set)
    killed_monsters: Set[Pos] = field(default_factory=set)
    pressed_buttons: Set[Pos] = field(default_factory=set)

    # 失败检测
    last_action: int = ACTION_NOOP
    stuck_count: int = 0

    def reset(self, task_id: Optional[str] = None):
        self.task_id = task_id
        self.step = 0
        self.last_player = None
        self.facing = "up"
        self.has_key = False
        self.has_sword = False
        self.keys = 0
        self.gold = 0
        self.items.clear()
        self.tools.clear()
        self.opened_chests.clear()
        self.killed_monsters.clear()
        self.pressed_buttons.clear()
        self.last_action = ACTION_NOOP
        self.stuck_count = 0

    def update(self, sym: SymbolicObs, info=None):
        self.step += 1

        # 只把 info 当作兼容接口。最终不要读隐藏状态。
        # 目前允许谨慎读取 inventory，因为项目说明中物品栏可作为显式输入。
        inv = None
        if isinstance(info, dict):
            inv = info.get("inventory", None)

        if inv:
            old_keys = self.keys
            old_gold = self.gold
            old_items = set(self.items)
            old_tools = set(self.tools)

            self.keys = int(inv.get("keys", 0))
            self.gold = int(inv.get("gold", 0))
            self.items = set(inv.get("items", []))
            self.tools = set(inv.get("tools", []))

            self.has_key = self.keys > 0
            self.has_sword = ("sword" in self.tools) or ("sword" in self.items) or (inv.get("equipped", {}).get("A") == "sword")

            if self.keys > old_keys:
                print(f"[LOOT] step={self.step} got KEY: {old_keys} -> {self.keys}")

            if self.gold > old_gold:
                print(f"[LOOT] step={self.step} got GOLD: {old_gold} -> {self.gold}")

            new_items = self.items - old_items
            if new_items:
                print(f"[LOOT] step={self.step} got ITEM: {new_items}")

            new_tools = self.tools - old_tools
            if new_tools:
                print(f"[LOOT] step={self.step} got TOOL: {new_tools}")

        # 更新 facing：根据玩家 tile 变化推断
        if self.last_player is not None and sym.player is not None:
            lx, ly = self.last_player
            x, y = sym.player
            if x > lx:
                self.facing = "right"
            elif x < lx:
                self.facing = "left"
            elif y > ly:
                self.facing = "down"
            elif y < ly:
                self.facing = "up"

            # 卡住检测
            if sym.player == self.last_player and self.last_action in {
                ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
            }:
                self.stuck_count += 1
            else:
                self.stuck_count = 0

        self.last_player = sym.player
        sym.facing = self.facing

        if isinstance(info, dict):
            events = info.get("events", {})
            flags = events.get("flags", {}) if isinstance(events, dict) else {}
            details = events.get("details", []) if isinstance(events, dict) else []

            interesting = [
                "chest_opened",
                "key_collected",
                "gold_collected",
                "item_collected",
                "agent_healed",
                "door_opened",
                "room_changed",
                "world_completed",
            ]

            happened = [name for name in interesting if flags.get(name, False)]

            if happened:
                print(
                    f"[EVENT] step={self.step}",
                    "happened=", happened,
                    "details=", details,
                )


class PixelPerception:
    def __init__(self):
        self.static_clf = StaticTileClassifier()
        self.player_detector = PlayerDetector()
        self.monster_detector = MonsterDetector()
        self.exit_detector = ExitDetector()

    def __call__(self, obs):
        frame = np.asarray(obs)

        # 防止误传 render()，只取地图区域
        frame = frame[:128, :160, :3]

        grid = np.zeros((8, 10), dtype=np.int64)

        # 1. 静态 tile 分类
        for y in range(8):
            for x in range(10):
                patch = frame[y*16:(y+1)*16, x*16:(x+1)*16, :]
                label, name, score = self.static_clf.classify_tile(patch)

                # 这里阈值可以设严格一点。
                # 如果 score 太大，默认 floor，避免误判。
                if score < 500:
                    grid[y, x] = label
                else:
                    grid[y, x] = EMPTY

        exit_infos = self.exit_detector.detect(frame)

        exits = []
        for e in exit_infos:
            x, y = e["tile"]
            exits.append((x, y))
            grid[y, x] = EXIT

        # 2. 玩家覆盖修正
        player_info = self.player_detector.detect(frame)
        player = None
        facing = "down"

        if player_info is not None:
            player = player_info["tile"]
            facing = player_info["facing"]
            px, py = player
            grid[py, px] = PLAYER

        # 3. 怪物覆盖修正
        monsters = []
        for m in self.monster_detector.detect_all(frame):
            tx, ty = m["tile"]
            monsters.append((tx, ty))
            grid[ty, tx] = MONSTER

        return self.grid_to_symbolic(grid, player, facing, monsters, exits)

    def grid_to_symbolic(self, grid, player, facing, monsters, exits_hint=None):
        chests = []
        traps = []
        buttons = []
        switches = []
        npcs = []
        gaps = []
        bridges = []
        exits = list(exits_hint) if exits_hint is not None else []

        for y in range(8):
            for x in range(10):
                v = int(grid[y, x])

                if v == CHEST:
                    chests.append((x, y))
                elif v == TRAP:
                    traps.append((x, y))
                elif v == BUTTON:
                    buttons.append((x, y))
                elif v == SWITCH:
                    switches.append((x, y))
                elif v == NPC:
                    npcs.append((x, y))
                elif v == GAP:
                    gaps.append((x, y))
                elif v == BRIDGE:
                    bridges.append((x, y))
                elif v == EXIT:
                    p = (x, y)
                    if p not in exits:
                        exits.append((x, y))

        return SymbolicObs(
            grid=grid,
            player=player,
            facing=facing,
            monsters=monsters,
            chests=chests,
            exits=exits,
            traps=traps,
            buttons=buttons,
            switches=switches,
        )
    
@dataclass
class Subgoal:
    kind: str
    target: Optional[Pos] = None


class SymbolicPlanner:
    def next_subgoal(self, sym: SymbolicObs, belief: BeliefState) -> Subgoal:
        """
        上层 planner：决定现在应该干什么。
        先实现 Task 1/通用逻辑：
        1. 没钥匙 -> 找宝箱
        2. 有钥匙 -> 找出口
        """

        # 玩家位置识别失败时，不要乱动
        if sym.player is None:
            return Subgoal("wait")

        # 没钥匙：优先去最近宝箱
        if not belief.has_key:
            chest = self.nearest(sym.player, sym.chests)
            if chest is not None:
                return Subgoal("find_chest", chest)
            return Subgoal("explore")
        

        # 有钥匙：去出口
        exit_pos = self.nearest(sym.player, sym.exits)
        if exit_pos is not None:
            return Subgoal("go_exit", exit_pos)

        return Subgoal("explore")

    def nearest(self, start: Pos, candidates: List[Pos]) -> Optional[Pos]:
        if not candidates:
            return None
        sx, sy = start
        return min(candidates, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))

    # def exit_direction_from_tile(self, exit_pos: Pos) -> int:
    #     x, y = exit_pos
    #     if y == 0:
    #         return ACTION_UP
    #     if y == ROOM_H - 1:
    #         return ACTION_DOWN
    #     if x == 0:
    #         return ACTION_LEFT
    #     if x == ROOM_W - 1:
    #         return ACTION_RIGHT
    #     return ACTION_NOOP
    

def neighbors(p: Pos) -> List[Tuple[Pos, int]]:
    x, y = p
    return [
        ((x, y - 1), ACTION_UP),
        ((x, y + 1), ACTION_DOWN),
        ((x - 1, y), ACTION_LEFT),
        ((x + 1, y), ACTION_RIGHT),
    ]


def in_bounds(p: Pos) -> bool:
    x, y = p
    return 0 <= x < ROOM_W and 0 <= y < ROOM_H


def is_passable(tile: int) -> bool:
    # 宝箱、墙、怪物、陷阱、gap 暂时都不走
    return tile in {EMPTY, PLAYER, EXIT, BUTTON, BRIDGE, SWITCH}


def bfs_path(grid: np.ndarray, start: Pos, goal: Pos) -> List[int]:
    """
    返回 tile 级动作序列，比如 [RIGHT, RIGHT, UP]
    """
    if start == goal:
        return []

    q = deque([start])
    parent: Dict[Pos, Tuple[Optional[Pos], Optional[int]]] = {
        start: (None, None)
    }

    while q:
        cur = q.popleft()

        for nxt, act in neighbors(cur):
            if not in_bounds(nxt):
                continue
            if nxt in parent:
                continue

            x, y = nxt
            if not is_passable(int(grid[y, x])):
                continue

            parent[nxt] = (cur, act)

            if nxt == goal:
                # 回溯动作
                actions = []
                p = nxt
                while parent[p][0] is not None:
                    prev, a = parent[p]
                    actions.append(a)
                    p = prev
                actions.reverse()
                return actions

            q.append(nxt)

    return []


def repeat_action(action: int, n: int) -> List[int]:
    return [action] * n


def expand_tile_actions(tile_actions: List[int]) -> List[int]:
    pixel_actions = []
    for a in tile_actions:
        pixel_actions.extend(repeat_action(a, TILE_SIZE))
    return pixel_actions


def adjacent_tiles(pos: Pos) -> List[Pos]:
    x, y = pos
    return [
        (x, y - 1),
        (x, y + 1),
        (x - 1, y),
        (x + 1, y),
    ]


def action_to_face(src: Pos, dst: Pos) -> int:
    sx, sy = src
    dx, dy = dst
    if dx > sx:
        return ACTION_RIGHT
    if dx < sx:
        return ACTION_LEFT
    if dy > sy:
        return ACTION_DOWN
    if dy < sy:
        return ACTION_UP
    return ACTION_NOOP

class OptionController:
    def build_actions(
        self,
        sym: SymbolicObs,
        belief: BeliefState,
        subgoal: Subgoal
    ) -> List[int]:

        if sym.player is None:
            return [ACTION_NOOP]

        if subgoal.kind == "wait":
            return [ACTION_NOOP]

        if subgoal.kind == "find_chest" and subgoal.target is not None:
            return self.actions_to_interactable(sym, subgoal.target)

        if subgoal.kind == "go_exit" and subgoal.target is not None:
            return self.actions_to_exit(sym, subgoal.target)

        if subgoal.kind == "explore":
            # 最简单探索：先等一下，后面再做 frontier exploration
            return [ACTION_NOOP]

        return [ACTION_NOOP]

    def actions_to_exit(self, sym: SymbolicObs, exit_pos: Pos) -> List[int]:
        assert sym.player is not None

        # 先走到出口 tile
        tile_actions = bfs_path(sym.grid, sym.player, exit_pos)
        actions = expand_tile_actions(tile_actions)

        # 再朝边界方向多走 32 步
        out_action = self.exit_direction_from_tile(exit_pos)
        if out_action != ACTION_NOOP:
            actions.extend([out_action] * 32)

        return actions

    def exit_direction_from_tile(self, exit_pos: Pos) -> int:
        x, y = exit_pos
        if y == 0:
            return ACTION_UP
        if y == ROOM_H - 1:
            return ACTION_DOWN
        if x == 0:
            return ACTION_LEFT
        if x == ROOM_W - 1:
            return ACTION_RIGHT
        return ACTION_NOOP

    def actions_to_interactable(self, sym: SymbolicObs, obj_pos: Pos) -> List[int]:
        """
        去到物体相邻格，然后面向物体，按 A。
        适用于 chest / switch / NPC。
        """
        assert sym.player is not None

        candidates = []
        for p in adjacent_tiles(obj_pos):
            if not in_bounds(p):
                continue
            x, y = p
            if is_passable(int(sym.grid[y, x])):
                candidates.append(p)

        if not candidates:
            return [ACTION_NOOP]

        # 选距离玩家最近的相邻格
        px, py = sym.player
        target_adj = min(
            candidates,
            key=lambda p: abs(p[0] - px) + abs(p[1] - py)
        )

        tile_actions = bfs_path(sym.grid, sym.player, target_adj)
        actions = expand_tile_actions(tile_actions)

        # 到达相邻格后，移动一步方向键让角色朝向宝箱
        face_action = action_to_face(target_adj, obj_pos)
        if face_action != ACTION_NOOP:
            actions.append(face_action)

        # 按 A
        actions.append(ACTION_A)
        return actions


class SafetyShield:
    def filter(self, action: int, sym: SymbolicObs, belief: BeliefState) -> int:
        if sym.player is None:
            return ACTION_NOOP

        if action in {ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT}:
            if self.is_exit_leaving_action(sym.player, action, sym.exits):
                return action

            nxt = self.predict_next_tile(sym.player, action)

            if not in_bounds(nxt):
                return ACTION_NOOP

            x, y = nxt
            tile = int(sym.grid[y, x])

            # 不主动走进墙、陷阱、gap、怪物
            if tile in {WALL, TRAP, GAP, MONSTER}:
                return ACTION_NOOP

        return action

    def predict_next_tile(self, pos: Pos, action: int) -> Pos:
        x, y = pos
        if action == ACTION_UP:
            return (x, y - 1)
        if action == ACTION_DOWN:
            return (x, y + 1)
        if action == ACTION_LEFT:
            return (x - 1, y)
        if action == ACTION_RIGHT:
            return (x + 1, y)
        return pos
    
    def is_exit_leaving_action(self, pos: Pos, action: int, exits: List[Pos]) -> bool:
        x, y = pos
        
        if pos not in exits:
            return False
        

        return (
            (y == 0 and action == ACTION_UP) or
            (y == ROOM_H - 1 and action == ACTION_DOWN) or
            (x == 0 and action == ACTION_LEFT) or
            (x == ROOM_W - 1 and action == ACTION_RIGHT)
        )
    
class Policy:
    def __init__(self) -> None:
        self.perception = PixelPerception()
        self.belief = BeliefState()
        self.planner = SymbolicPlanner()
        self.controller = OptionController()
        self.shield = SafetyShield()

        self.action_queue: Deque[int] = deque()
        self.current_subgoal: Optional[Subgoal] = None

        self.last_sym: Optional[SymbolicObs] = None
        self.perception_interval = 4   # 先用 4，稳定后可以改成 8

        self.force_exit_action: Optional[int] = None
        self.force_exit_steps: int = 0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed
        self.belief.reset(task_id=task_id)
        self.action_queue.clear()
        self.current_subgoal = None

        self.last_sym = None
        self.force_exit_action = None
        self.force_exit_steps = 0

    
    def act(self, obs, info=None) -> int:

        # 已经进入强制出门模式：不要识图，不要 shield，直接往外走
        if self.force_exit_steps > 0 and self.force_exit_action is not None:
            self.force_exit_steps -= 1

            if self.force_exit_steps % 10 == 0:
                print(
                    "[FORCE_EXIT]",
                    "action=", self.force_exit_action,
                    "steps_left=", self.force_exit_steps,
                )

            return int(self.force_exit_action)
        
        need_vision = (
        self.last_sym is None
        or not self.action_queue
        or self.belief.step % self.perception_interval == 0
        )

        if need_vision:
            sym = self.perception(obs)
            self.last_sym = sym
            self.belief.update(sym, info)
        else:
            sym = self.last_sym
            self.belief.step += 1

        replanned = False

        # 3. 判断是否需要重新规划
        if self.need_replan(sym, info):
            replanned = True
            self.current_subgoal = self.planner.next_subgoal(sym, self.belief)
            actions = self.controller.build_actions(
                sym,
                self.belief,
                self.current_subgoal
            )
            self.action_queue = deque(actions)

            # 调试输出 1：只在重新规划时打印
            print(
                "[REPLAN]",
                "step=", self.belief.step,
                "player=", sym.player,
                "chests=", sym.chests,
                "exits=", sym.exits,
                "monsters=", sym.monsters,
                "has_key=", self.belief.has_key,
                "subgoal=", self.current_subgoal,
                "new_queue=", len(self.action_queue),
            )

        # 4. 没动作就等待
        if not self.action_queue:
            raw_action = ACTION_NOOP
        else:
            raw_action = self.action_queue.popleft()

        # 如果当前已经在出口边缘，并且队列动作正是朝外走，
        # 进入强制出门模式，绕过 shield。
        exit_action = self.exit_action_if_at_exit(sym)

        if (
            self.current_subgoal is not None
            and self.current_subgoal.kind == "go_exit"
            and exit_action is not None
            and raw_action == exit_action
        ):
            self.force_exit_action = exit_action
            self.force_exit_steps = 40

            print(
                "[START_FORCE_EXIT]",
                "step=", self.belief.step,
                "player=", sym.player,
                "exits=", sym.exits,
                "action=", exit_action,
            )

            return int(exit_action)


        # 5. 安全过滤
        action = self.shield.filter(raw_action, sym, self.belief)

        if self.belief.step % 10 == 0 or replanned:
            print(
                "[ACT]",
                "step=", self.belief.step,
                "player=", sym.player,
                "raw=", raw_action,
                "safe=", action,
                "queue_left=", len(self.action_queue),
                "stuck=", self.belief.stuck_count,
                "subgoal=", self.current_subgoal,
            )

        self.belief.last_action = action
        return int(action)

    def need_replan(self, sym: SymbolicObs, info=None) -> bool:
        # 没有动作了，必须重新规划
        if not self.action_queue:
            return True

        # 识别不到玩家，先不继续盲走
        if sym.player is None:
            self.action_queue.clear()
            return True

        # # 卡住了，重新规划
        # if self.belief.stuck_count >= 4:
        #     self.action_queue.clear()
        #     return True

        # # reward / info 里如果出现关键事件，也重新规划
        # # 注意：这里只建议用于训练/调试；最终要保证不使用隐藏状态。
        # if isinstance(info, dict):
        #     events = info.get("events", {})
        #     flags = events.get("flags", {}) if isinstance(events, dict) else {}

        #     important = [
        #         "chest_opened",
        #         "key_collected",
        #         "item_collected",
        #         "monster_killed",
        #         "door_opened",
        #         "button_pressed",
        #         "switch_activated",
        #         "bridge_rotated",
        #         "room_changed",
        #         "world_completed",
        #         "action_blocked",
        #     ]

        #     for name in important:
        #         if flags.get(name, False):
        #             self.action_queue.clear()
        #             return True

        return False
    
    def exit_action_if_at_exit(self, sym: SymbolicObs) -> Optional[int]:
        """
        如果玩家已经在出口边缘，返回应该朝哪个方向出门。
        允许出口检测只返回双格门的其中一个 tile。
        """
        if sym.player is None:
            return None

        px, py = sym.player

        for ex, ey in sym.exits:
            # north exit
            if ey == 0 and py == 0 and abs(px - ex) <= 1:
                return ACTION_UP

            # south exit
            if ey == ROOM_H - 1 and py == ROOM_H - 1 and abs(px - ex) <= 1:
                return ACTION_DOWN

            # west exit
            if ex == 0 and px == 0 and abs(py - ey) <= 1:
                return ACTION_LEFT

            # east exit
            if ex == ROOM_W - 1 and px == ROOM_W - 1 and abs(py - ey) <= 1:
                return ACTION_RIGHT

        return None


def make_policy() -> Policy:
    return Policy()