"""
sandbox/make_demo_set.py — Build a small, *curated* demo subset from the full dataset.

A purely random sample of 100K candidates is dominated by poor fits, so a live
demo would show almost nothing interesting at the honeypot / disqualifier /
reranking stages. This script picks a deliberately mixed subset:

  - strong fits      : top candidates by structured JD-fit score (clean profiles)
  - honeypots        : profiles the honeypot detector flags
  - disqualified     : profiles hitting >= 1 JD disqualifier (consulting-only, zombie, ...)
  - random           : uniform background sample (realistic base rate)

NOTE: the subset is *selected using our own scorer*, so it must never be used to
evaluate ranking quality — it is a demo prop only.

Usage:
  python sandbox/make_demo_set.py --src data/candidates.jsonl --out data/demo_candidates.jsonl
"""

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.structured_scorer import compute_structured_score
from src.honeypot_detector import detect_honeypot
from src.disqualifiers import compute_disqualifier_penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "candidates.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "demo_candidates.jsonl"))
    ap.add_argument("--n-strong", type=int, default=60)
    ap.add_argument("--n-honeypot", type=int, default=20)
    ap.add_argument("--n-disq", type=int, default=30)
    ap.add_argument("--n-random", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    today = date.today()

    strong, honeypots, disq, lines = [], [], [], []
    with open(args.src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            c = json.loads(line)
            hp, _ = detect_honeypot(c)
            dq, _ = compute_disqualifier_penalty(c, today)
            if hp < 1.0:
                honeypots.append(i)
            elif dq < 1.0:
                disq.append(i)
            else:
                s, _ = compute_structured_score(c, today)
                strong.append((s, i))
            if (i + 1) % 20000 == 0:
                print(f"  scanned {i + 1:,}")

    strong.sort(reverse=True)
    picked = [i for _, i in strong[: args.n_strong]]
    picked += rng.sample(honeypots, min(args.n_honeypot, len(honeypots)))
    picked += rng.sample(disq, min(args.n_disq, len(disq)))
    taken = set(picked)
    pool = [i for i in range(len(lines)) if i not in taken]
    picked += rng.sample(pool, args.n_random)
    rng.shuffle(picked)  # don't leak the "answer" through file order

    with open(args.out, "w", encoding="utf-8") as f:
        for i in picked:
            f.write(lines[i] + "\n")

    print(f"Scanned {len(lines):,} | honeypots={len(honeypots):,} disq={len(disq):,}")
    print(f"Wrote {len(picked)} candidates -> {args.out}")


if __name__ == "__main__":
    main()
