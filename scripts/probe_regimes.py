"""E074: decompose the held-out single-target loss into regimes (pure arithmetic, no GPU)."""
import json, pathlib
sel = json.loads(pathlib.Path("results/selectability.json").read_text(encoding="utf-8"))
rep = json.loads(pathlib.Path("results/gate_sweep/report.json").read_text(encoding="utf-8"))
frames = {r["sequence"]: int(r["n_report_frames"]) for r in rep["points"]["margin_0.20"]["rows"]}

for s in sel:
    s["frames"] = frames[s["seq"]]
    s["eff"] = s["tsr"] / s["presence"] if s["presence"] > 0 else 0.0
    if s["idc"] < 0.50 and s["eff"] < 0.80:
        s["regime"] = "churn"
    elif s["eff"] < 0.50:
        s["regime"] = "wrong-hold"
    else:
        s["regime"] = "working"

tot = sum(s["frames"] for s in sel)
pooled = sum(s["tsr"] * s["frames"] for s in sel) / tot
cap_all = sum(s["presence"] * s["frames"] for s in sel) / tot
print(f"held-out pooled TSR {pooled:.4f} over {tot} report frames; presence cap {cap_all:.4f}")
print(f"total loss {1 - pooled:.4f} = absence {1 - cap_all:.4f} + in-present {cap_all - pooled:.4f}\n")

print(f"{'regime':11s} {'n':>2s} {'frames':>6s} {'pooled TSR':>10s} {'presence':>8s} {'eff':>6s} {'loss vs cap':>11s}")
for name in ("working", "wrong-hold", "churn"):
    g = [s for s in sel if s["regime"] == name]
    if not g: continue
    f = sum(s["frames"] for s in g)
    t = sum(s["tsr"] * s["frames"] for s in g) / f
    p = sum(s["presence"] * s["frames"] for s in g) / f
    print(f"{name:11s} {len(g):2d} {f:6d} {t:10.4f} {p:8.4f} {t / p if p else 0:6.3f} {p - t:11.4f}")
    print("            " + ", ".join(f"{s['seq'][-4:]}({s['tsr']:.2f}/{s['idc']:.2f})" for s in g))

print("\nAttribution of the in-present loss by regime (share of the total 0.79):")
in_present = cap_all - pooled
for name in ("working", "wrong-hold", "churn"):
    g = [s for s in sel if s["regime"] == name]
    if not g: continue
    f = sum(s["frames"] for s in g)
    t = sum(s["tsr"] * s["frames"] for s in g) / f
    p = sum(s["presence"] * s["frames"] for s in g) / f
    print(f"  {name:11s} {(p - t) * f / tot:+.4f} of the {in_present:.4f} in-present loss")
pathlib.Path("results/selectability_regimes.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
