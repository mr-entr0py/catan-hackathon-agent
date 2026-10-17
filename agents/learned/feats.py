"""Observable-only position features from one player's point of view (no hidden-info leakage).

Opponent's hidden dev cards are NOT used: only their count. Our own hand/devs are known.
"""
from catanatron.features import build_production_features, reachability_features
from catanatron.models.enums import RESOURCES

PROD = build_production_features(True)
DEVS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]


def phi(game, me):
    st = game.state
    ps = st.player_state
    i = st.color_to_index[me]
    j = 1 - i
    opp = st.colors[j]
    prod = PROD(game, me)
    f = []
    for P in ("P0", "P1"):
        v = [prod[f"EFFECTIVE_{P}_{r}_PRODUCTION"] for r in RESOURCES]
        f += v
        f.append(sum(v))
        f.append(sum(x > 0 for x in v))
    rs = reachability_features(game, me, 2)
    f += [rs[f"P0_1_ROAD_REACHABLE_{r}"] for r in RESOURCES]
    f.append(sum(rs[f"P0_2_ROAD_REACHABLE_{r}"] for r in RESOURCES))
    f.append(sum(rs[f"P1_1_ROAD_REACHABLE_{r}"] for r in RESOURCES))
    board = st.board
    f.append(len(board.buildable_node_ids(me)))
    f.append(len(board.buildable_node_ids(opp)))
    for k, c in ((i, me), (j, opp)):
        p = f"P{k}_"
        hand = [ps[p + r + "_IN_HAND"] for r in RESOURCES]
        n = sum(hand)
        f += [ps[p + "VICTORY_POINTS"], ps[p + "LONGEST_ROAD_LENGTH"], ps[p + "HAS_ROAD"], ps[p + "HAS_ARMY"],
              ps[p + "PLAYED_KNIGHT"], ps[p + "ROADS_AVAILABLE"], ps[p + "SETTLEMENTS_AVAILABLE"],
              ps[p + "CITIES_AVAILABLE"]]
        f += hand
        f += [n, float(n > 7)]
        f.append(sum(ps[p + d + "_IN_HAND"] for d in DEVS))      # dev count (public for opp)
        ports = board.get_player_port_resources(c)
        f += [float(r in ports) for r in RESOURCES] + [float(None in ports)]
    p = f"P{i}_"
    f += [ps[p + "VICTORY_POINT_IN_HAND"], ps[p + "KNIGHT_IN_HAND"],
          ps[p + "YEAR_OF_PLENTY_IN_HAND"] + ps[p + "MONOPOLY_IN_HAND"] + ps[p + "ROAD_BUILDING_IN_HAND"]]
    t = st.num_turns / 100.0
    vd = ps[f"P{i}_VICTORY_POINTS"] - ps[f"P{j}_VICTORY_POINTS"]
    f += [t, float(st.current_color() == me), vd, vd * t,
          ps[f"P{i}_ACTUAL_VICTORY_POINTS"], float(st.colors[0] == me)]
    return f
