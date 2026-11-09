"""Command line entry point.

  python -m clq.cli eval [--mode mock|litellm]    run every policy, write reports/ and dashboard/data.json
  python -m clq.cli simulate [--volume N] [...]    print the economics comparison for current config
  python -m clq.cli label --policy cascade+cache   grade delivered replies by hand -> data/human_labels.csv
  python -m clq.cli calibrate                      kappa of the judge against your labels
"""
from __future__ import annotations

import argparse
import csv
import json

from .util import ROOT, load_cases, load_config


def cmd_eval(args):
    from .evaluate import Harness
    from .reports import write_all
    cfg, cases = load_config(args.config), load_cases()
    res = Harness(cfg, cases, mode=args.mode).run()
    write_all(res, cfg, cases)
    print((ROOT / "reports" / "summary.md").read_text())


def cmd_simulate(args):
    from .economics import compare, default_params
    p = default_params(load_config(args.config))
    for k in ("volume", "human_cost", "silent_cost", "recall", "false_flag", "cache_hit_easy"):
        v = getattr(args, k)
        if v is not None:
            p[k] = v
    print(f"{'policy':16s} {'pass':>7s} {'silent':>7s} {'handoff':>7s} {'model/1k':>9s} {'full/1k':>9s} {'p50':>6s} {'p90':>6s} {'p95':>6s} {'p99':>6s}")
    for pol, r in compare(p).items():
        l = r["latency_ms"]
        print(f"{pol:16s} {r['pass_rate']:7.1%} {r['silent_error_rate']:7.1%} {r['handoff_rate']:7.1%} "
              f"{r['model_cost_per_1k']:9.3f} {r['fully_loaded_per_1k']:9.1f} {l['p50']:6.0f} {l['p90']:6.0f} {l['p95']:6.0f} {l['p99']:6.0f}")


def cmd_label(args):
    path = ROOT / "reports" / "records.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    recs = [r for r in recs if r["policy"] == args.policy and r["category"] != "HANDOFF"]
    cases = {c["case_id"]: c for c in load_cases()}
    out = ROOT / "data" / "human_labels.csv"
    done = set()
    if out.exists():
        done = {(r["policy"], r["case_id"]) for r in csv.DictReader(out.open())}
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, ["policy", "case_id", "faithfulness_pass", "judge_faithfulness_pass", "note"])
        if new:
            w.writeheader()
        for r in recs:
            if (r["policy"], r["case_id"]) in done:
                continue
            c = cases[r["case_id"]]
            print("\n" + "=" * 80 + f"\n{r['case_id']}  [{c['stratum']}]\nQUERY: {c['input']['user_query']}")
            for ch in c["input"]["context_chunks"]:
                print(f"  [{ch['id']}] {ch['text']}")
            print(f"REPLY: {r['reply']}")
            ans = input("Every claim grounded in the context? [y/n/q]: ").strip().lower()
            if ans == "q":
                break
            w.writerow({"policy": r["policy"], "case_id": r["case_id"], "faithfulness_pass": int(ans == "y"),
                        "judge_faithfulness_pass": int(r["judge"]["factual_faithfulness"] == 2), "note": ""})


def cmd_calibrate(args):
    from .stats import kappa_with_ci
    rows = list(csv.DictReader((ROOT / "data" / "human_labels.csv").open()))
    h = [int(r["faithfulness_pass"]) for r in rows]
    j = [int(r["judge_faithfulness_pass"]) for r in rows]
    res = kappa_with_ci(h, j)
    slo = load_config(args.config)["slo"]
    res["gate_passed"] = res["kappa"] >= slo["min_judge_kappa_point"] and res["ci_low"] >= slo["min_judge_kappa_ci_low"]
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(prog="clq")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--mode", choices=["mock", "litellm"], default="mock")
    s = sub.add_parser("simulate")
    for k, t in (("volume", int), ("human_cost", float), ("silent_cost", float), ("recall", float),
                 ("false_flag", float), ("cache_hit_easy", float)):
        s.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t)
    l = sub.add_parser("label")
    l.add_argument("--policy", default="cascade+cache")
    sub.add_parser("calibrate")
    args = ap.parse_args()
    {"eval": cmd_eval, "simulate": cmd_simulate, "label": cmd_label, "calibrate": cmd_calibrate}[args.cmd](args)


if __name__ == "__main__":
    main()
