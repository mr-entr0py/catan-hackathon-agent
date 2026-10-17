"""larpmaxxers v2: F-style 1-ply greedy with an extended, tunable value function.

Override settings for experiments with env CATAN_CFG='{"hand_devs": 1e5}'.
"""
import json
import os
import random
import time

from catanatron.features import build_production_features, reachability_features
from catanatron.models.enums import ActionType, RESOURCES, SETTLEMENT, CITY
from catanatron.state_functions import (get_longest_road_length, get_played_dev_cards,
                                        player_key, player_num_dev_cards,
                                        player_num_resource_cards)

AT = ActionType
PROD = build_production_features(True)
PROBA_POINT = 2.778 / 100

CFG = {
    "public_vps": 3e14, "production": 1e8, "enemy_production": -1e8, "num_tiles": 1,
    "reachable_production_0": 0, "reachable_production_1": 1e4, "buildable_nodes": 1e3,
    "longest_road": 10, "lr_always": 0.0, "hand_synergy": 1e2, "hand_resources": 1,
    "discard_penalty": -5, "hand_devs": 10, "army_size": 10.1, "variety": 4.0,
    "port2": 0.0, "port3": 0.0, "lookahead": 0,
}
CFG.update({"lr_always": 1e6, "discard_penalty": -1e3})
CFG.update(json.loads(os.environ.get("CATAN_CFG", "{}")))


def value(game, color, p=CFG):
    st = game.state
    s = PROD(game, color)
    mine = [s[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES]
    opp = [s[f"EFFECTIVE_P1_{r}_PRODUCTION"] for r in RESOURCES]
    production = sum(mine) + sum(m != 0 for m in mine) * p["variety"] * PROBA_POINT
    enemy_production = sum(opp)

    key = player_key(st, color)
    ps = st.player_state
    rs = reachability_features(game, color, 2)
    r0 = sum(rs[f"P0_0_ROAD_REACHABLE_{r}"] for r in RESOURCES)
    r1 = sum(rs[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES)

    wheat, ore, sheep = ps[f"{key}_WHEAT_IN_HAND"], ps[f"{key}_ORE_IN_HAND"], ps[f"{key}_SHEEP_IN_HAND"]
    brick, wood = ps[f"{key}_BRICK_IN_HAND"], ps[f"{key}_WOOD_IN_HAND"]
    d_city = (max(2 - wheat, 0) + max(3 - ore, 0)) / 5.0
    d_sett = (max(1 - wheat, 0) + max(1 - sheep, 0) + max(1 - brick, 0) + max(1 - wood, 0)) / 4.0
    synergy = (2 - d_city - d_sett) / 2

    n_hand = player_num_resource_cards(st, color)
    discard = p["discard_penalty"] if n_hand > 7 else 0

    b = st.buildings_by_color[color]
    tiles = set()
    for n in b[SETTLEMENT] + b[CITY]:
        tiles.update(st.board.map.adjacent_tiles[n])
    n_buildable = len(st.board.buildable_node_ids(color))
    lr = get_longest_road_length(st, color)
    lr_factor = p["longest_road"] if n_buildable == 0 else 0.1

    v = (ps[f"{key}_VICTORY_POINTS"] * p["public_vps"]
         + production * p["production"] + enemy_production * p["enemy_production"]
         + r0 * p["reachable_production_0"] + r1 * p["reachable_production_1"]
         + synergy * p["hand_synergy"] + n_buildable * p["buildable_nodes"]
         + len(tiles) * p["num_tiles"] + n_hand * p["hand_resources"] + discard
         + lr * lr_factor + lr * p["lr_always"]
         + player_num_dev_cards(st, color) * p["hand_devs"]
         + get_played_dev_cards(st, color, "KNIGHT") * p["army_size"])
    if p["port2"] or p["port3"]:
        ports = st.board.get_player_port_resources(color)
        for i, r in enumerate(RESOURCES):
            if r in ports:
                v += p["port2"] * mine[i]
        if None in ports:
            v += p["port3"] * sum(mine)
    return float(v)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions):
        try:
            return self._decide(obs, playable_actions)
        except Exception:
            return 0

    def _decide(self, obs, playable_actions):
        me = obs.me
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, a in enumerate(playable_actions):
            s = world.copy()
            try:
                s.execute(a)
            except Exception:
                continue
            if s.winning_color() == me:
                return i
            v = value(s, me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i
