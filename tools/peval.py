"""Parallel version of catan_kit.evaluate: splits board pairs across worker processes.

    python tools/peval.py --a agents/v1/agent.py --b examples/f_agent.py --games 400 --jobs 4

Same seeds/seat pairing as the official script (seed0 + i//2), so results are comparable.
Appends one JSON line per run to results/runs.jsonl.
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catan_kit._patch import ensure_hash_seed


def _chunk(args):
    a, b, seeds, idx = args
    from catan_kit import match
    recs, info = match.play_chunk(a, b, seeds, idx)
    return recs


def main():
    ensure_hash_seed()
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seed0", type=int, default=990000)
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--tag", default="")
    args = p.parse_args()
    from catan_kit.evaluate import summarise
    n = args.games - args.games % 2
    seeds = [args.seed0 + i // 2 for i in range(n)]
    pairs = list(range(0, n, 2))
    k = max(1, args.jobs)
    jobs = []
    for j in range(k):
        mine = pairs[j::k]
        idx = [i for pi in mine for i in (pi, pi + 1)]
        if idx:
            jobs.append((args.a, args.b, [seeds[i] for i in idx], idx))
    t0 = time.time()
    with ProcessPoolExecutor(k) as ex:
        recs = [r for out in ex.map(_chunk, jobs) for r in out]
    res = summarise(recs)
    resb = summarise(recs, "b")
    res.update(wall_s=round(time.time() - t0, 1), a=args.a, b=args.b, seed0=args.seed0, tag=args.tag,
               b_mean_cpu_s=resb["mean_cpu_s"], b_over=resb["games_over_budget"],
               mean_turns=sum(r["turns"] for r in recs) / len(recs),
               first_seat_wr=sum(r["winner"] == "a" for r in recs if r["a_first"]) / max(1, sum(r["a_first"] for r in recs)))
    os.makedirs("results", exist_ok=True)
    with open("results/runs.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"{args.a} vs {args.b}: WR {res['win_rate']:.3f} +-{res['ci95']:.3f} ({res['wins']}/{res['games']}) "
          f"noWin {res['no_winner']} | cpu mean {res['mean_cpu_s']*1000:.0f}ms worst {res['worst_cpu_s']*1000:.0f}ms "
          f"over {res['games_over_budget']} err {res['errors']} to/crash {res['timeouts']}/{res['crashes']} "
          f"| asFirst {res['first_seat_wr']:.3f} | {res['wall_s']}s")


if __name__ == "__main__":
    main()
