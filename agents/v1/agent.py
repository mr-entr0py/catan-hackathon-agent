"""larpmaxxers v1: F's value function + 2-step lookahead within our own turn, under a CPU budget."""
import random
import time

from catanatron.models.enums import ActionType
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn

AT = ActionType
VALUE_FN = base_fn(DEFAULT_WEIGHTS)
# Actions after which the game is still in our hands with a deterministic result.
EXPANDABLE = {AT.BUILD_ROAD, AT.BUILD_SETTLEMENT, AT.BUILD_CITY, AT.MARITIME_TRADE,
              AT.PLAY_YEAR_OF_PLENTY, AT.PLAY_MONOPOLY, AT.PLAY_ROAD_BUILDING}
TOPK = 4


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions):
        t0 = time.process_time()
        left = obs.cpu_left if obs.cpu_left is not None else 1.0
        budget = max(0.002, 0.45 * (left - 0.04))
        deadline = t0 + budget
        me = obs.me
        world = obs.sample_world(self.rng)

        scored = []
        for i, a in enumerate(playable_actions):
            s = world.copy()
            try:
                s.execute(a)
            except Exception:
                continue
            if s.winning_color() == me:
                return i
            scored.append((VALUE_FN(s, me), i, a, s))
        if not scored:
            return 0
        scored.sort(key=lambda x: -x[0])
        best_v, best_i = scored[0][0], scored[0][1]

        # Second step: from the top candidates that keep the turn with us, what is the best follow-up?
        for v, i, a, s in scored[:TOPK]:
            if a.action_type not in EXPANDABLE or s.state.current_color() != me:
                continue
            for b in s.playable_actions:
                if time.process_time() > deadline:
                    return best_i
                if b.action_type == AT.END_TURN:
                    continue
                s2 = s.copy()
                try:
                    s2.execute(b)
                except Exception:
                    continue
                if s2.winning_color() == me:
                    return i
                v2 = VALUE_FN(s2, me)
                if v2 > best_v:
                    best_v, best_i = v2, i
        return best_i
