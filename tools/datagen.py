"""Generate (features, win) rows from in-process games for fitting a value function.

    python tools/datagen.py --games 2000 --jobs 4 --out data/run1 --seed0 5000000
Policies: each seat is F (base_fn) or our v2-lr greedy, with eps random moves.
"""
import argparse, os, random, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "agents", "learned"))


def worker(args):
    seeds, eps, p_rec, out = args
    import catan_kit  # noqa
    from catanatron.game import Game
    from catanatron.models.player import Color, Player
    from catanatron.players.value import base_fn, DEFAULT_WEIGHTS
    sys.path.insert(0, os.path.join(ROOT, "agents", "v2"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("v2", os.path.join(ROOT, "upload", "v2lr", "agent.py"))
    v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
    from feats import phi
    F = base_fn(DEFAULT_WEIGHTS)

    class Greedy(Player):
        def __init__(self, color, fn, rng):
            super().__init__(color); self.fn = fn; self.rng = rng
        def decide(self, game, actions):
            if len(actions) == 1: return actions[0]
            if self.rng.random() < eps: return self.rng.choice(actions)
            best, bv = actions[0], None
            for a in actions:
                g = game.copy(); g.execute(a)
                v = self.fn(g, self.color)
                if bv is None or v > bv: best, bv = a, v
            return best

    X, Y = [], []
    for s in seeds:
        rng = random.Random(s)
        fns = [F if rng.random() < 0.5 else v2.value for _ in range(2)]
        random.seed(s)
        g = Game([Greedy(Color.RED, fns[0], rng), Greedy(Color.BLUE, fns[1], rng)], seed=s)
        rows = []
        while g.winning_color() is None and g.state.num_turns < 400:
            g.play_tick()
            if rng.random() < p_rec and not g.state.is_initial_build_phase:
                for c in g.state.colors:
                    rows.append((phi(g, c), c))
        w = g.winning_color()
        for f, c in rows:
            X.append(f); Y.append(1.0 if c == w else 0.0)
    np.savez_compressed(out, X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32))
    return len(Y)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=2000); p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed0", type=int, default=5000000); p.add_argument("--eps", type=float, default=0.08)
    p.add_argument("--prec", type=float, default=0.15); p.add_argument("--out", default="data/run1")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(a.seed0, a.seed0 + a.games))
    chunks = [(seeds[k::a.jobs * 4], a.eps, a.prec, os.path.join(a.out, f"part{k}.npz")) for k in range(a.jobs * 4)]
    with ProcessPoolExecutor(a.jobs) as ex:
        print("rows", sum(ex.map(worker, chunks)))
