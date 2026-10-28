"""E077-prep 2: how much of the held-out loss sits in RECOVERABLE blind frames?"""
import csv, glob, collections
BLIND = {"occluded_soft", "lost_short", "lost_long", "out_of_frame"}
TRUTHY = {"1", "True", "true"}
files = sorted(glob.glob("results/e076_states/runs/*.csv"))
tot = present = blind = blind_present = blind_present_long = 0
per = []
for f in files:
    recs = list(csv.DictReader(open(f, encoding="utf-8", newline="")))
    n = len(recs); bp = 0; bl = 0; pl = 0
    for r in recs:
        p = r.get("gt_present") in TRUTHY
        b = r.get("state") in BLIND
        present += p; blind += b
        if p and b:
            bp += 1
            if r.get("state") == "lost_long": pl += 1
    tot += n; blind_present += bp; blind_present_long += pl
    per.append((f.split("__")[0].split("/")[-1], n, bp / n if n else 0, pl / n if n else 0))
per.sort(key=lambda x: -x[2])
print(f"{'sequence':18s} {'n':>4s} {'blind&present':>13s} {'of which lost_long':>18s}")
for name, n, bp, pl in per:
    print(f"{name:18s} {n:4d} {bp:13.3f} {pl:18.3f}")
print(f"\nTOTAL frames {tot}; blind&present {blind_present} ({blind_present/tot:.3f}) "
      f"of which lost_long {blind_present_long} ({blind_present_long/tot:.3f})")
print("Upside bound: converting every blind&present frame to success would lift pooled TSR by "
      f"+{blind_present/tot:.3f} (to {0.2104 + blind_present/tot:.3f} over the full 22-seq set, "
      "modulo set differences here).")
