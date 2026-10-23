"""Request corpora for the fast_parity bench: recorded agent requests and taxonomy kev-4b requests, one body per line.

  /data/vllm/pipelines/.venv/bin/python scripts/kev_requests.py --out /data/florian.leprat/fast_parity

agent-4b.jsonl  Browser-agent action requests the harness sent to Kev (HARNESS_KEV_DUMP), sampled from
                kev_rl/data/kev_requests_base.jsonl: real page states, tasks and action options, unchanged.
kev-4b.jsonl    The pipelines' taxonomy.kev requests for the 150 classification gold products: the purchase gate
                (noul + kind, workflows.taxonomy.functions._kev_bought_anything) and the rerank choice
                (_kev_rerank). Prompts and criteria come from the pipelines code itself. The gold rows keep each
                candidate's path and name only, so a rerank option lacks the description and example products
                production adds: shorter than live, otherwise as sent.
"""

import argparse
import json
import os
import random
import sys

RECORDED = "/home/florian.leprat/custom_browser-handoff/kev_rl/data/kev_requests_base.jsonl"
GOLD = ["classification", "classification-v2", "classification-pool"]


def agent(path: str, n: int, rng: random.Random) -> list[dict]:
    rows = [json.loads(line)["request"] for line in open(path)]
    return rng.sample(rows, min(n, len(rows)))


def taxonomy(pipelines: str, n: int, rng: random.Random) -> list[dict]:
    os.chdir(pipelines)  # settings read config.toml next to the package
    sys.path.insert(0, pipelines)
    from workflows.taxonomy import functions as f  # noqa: PLC0415

    products = {}
    for name in GOLD:
        for line in open(f"{pipelines}/datasets/{name}/data.jsonl"):
            row = json.loads(line)
            products.setdefault(row["product_id"], row)
    kinds = {str(getattr(k, "value", k)): v for k, v in f._KEV_GATE_KINDS.items()}
    out = []
    for row in products.values():
        candidates = [{"node_path": c["path"], "name": c["name"]} for c in row["candidates"]]
        out.append(
            {
                "model": "kev-4b",
                "state": row["classifier_query"],
                "questions": {"category": {"type": "choice", "instructions": f._KEV_RERANK_INSTRUCTIONS, "criteria": f._kev_criteria(candidates)}},
            }
        )
        out.append(
            {
                "model": "kev-4b",
                "state": row["input_product_name"],
                "questions": {
                    "bought": {"type": "noul", "instructions": f._GATE_PROMPT},
                    "kind": {"type": "choice", "instructions": "What kind of thing does this invoice line record", "criteria": kinds},
                },
            }
        )
    return rng.sample(out, min(n, len(out)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recorded", default=RECORDED)
    ap.add_argument("--pipelines", default="/data/vllm/pipelines")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    for name, rows in [("agent-4b", agent(a.recorded, a.n, rng)), ("kev-4b", taxonomy(a.pipelines, a.n, rng))]:
        path = f"{a.out}/{name}.jsonl"
        with open(path, "w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(path, len(rows))


if __name__ == "__main__":
    main()
