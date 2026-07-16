#!/usr/bin/env python3
"""Validate an S32000 Experanto export dir (Phase 5 stop-condition gate). Plain-python (no mozaik/neo
needed for these checks — spikes.npy is float64, meta.yml/json are plain). Reuses the pilot's checks.

Usage:  python validate_export.py /ws/S32000_full/export/trial0   [n_stimuli_expected]

Checks: (1) responses/meta.yml end_time == screen/timestamps[-1]; (2) video frames == 35 ms;
(3) per-image 497 ms + 49 ms post-blank present; (4) modality/tier counts; (5) spike_indices CSR
well-formed (monotonic, N+1, last == len(spikes)); (6) sizes.
"""
import json, os, sys
import numpy as np
import yaml

ex = sys.argv[1]
n_expected = int(sys.argv[2]) if len(sys.argv) > 2 else None
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

print(f"Validating {ex}")
resp = yaml.safe_load(open(f"{ex}/responses/meta.yml"))
ts = np.load(f"{ex}/screen/timestamps.npy")
cm = json.load(open(f"{ex}/screen/combined_meta.json"))
spikes = np.load(f"{ex}/responses/spikes.npy")
si = np.asarray(resp["spike_indices"])

# 1. timeline alignment
check("timeline: responses.end_time == screen timestamps[-1]",
      abs(float(resp["end_time"]) - float(ts[-1])) < 1e-3,
      f"end_time={resp['end_time']} ts[-1]={ts[-1]:.3f}")
# 2. video frame duration 35 ms (dominant diff)
d = np.round(np.diff(ts), 4)
vals, cnts = np.unique(d, return_counts=True)
top = vals[np.argmax(cnts)]
check("video frame duration == 0.035 s (dominant)", abs(top - 0.035) < 1e-4, f"dominant diff={top}s")
# 3. per-image timing present
check("per-image ~0.497 s present", np.any(np.abs(vals - 0.497) < 2e-3))
check("post-blank ~0.049 s present", np.any(np.abs(vals - 0.049) < 2e-3))
# 4. modality / tier counts
from collections import Counter
mods = Counter(v["modality"] for v in cm.values())
tiers = Counter(v.get("tier") for v in cm.values() if v.get("modality") != "blank")
print(f"    modalities={dict(mods)} tiers={dict(tiers)}")
n_stim = mods.get("image", 0) + mods.get("video", 0)
if n_expected:
    check(f"stimulus count == {n_expected}", n_stim == n_expected, f"got {n_stim}")
# 5. CSR spike_indices
check("spike_indices monotonic non-decreasing", np.all(np.diff(si) >= 0))
check("spike_indices last == len(spikes)", int(si[-1]) == len(spikes), f"{int(si[-1])} vs {len(spikes)}")
check("spike_indices[0] == 0", int(si[0]) == 0)
print(f"    n_signals={resp.get('n_signals')} n_spikes={len(spikes)} end_time={resp['end_time']}s "
      f"screen_entries={len(cm)} ts_frames={len(ts)}")
# 6. sizes
for sub in ("responses", "screen"):
    p = f"{ex}/{sub}"
    tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs)
    print(f"    {sub}: {tot/1e9:.2f} GB")

print("VALIDATION:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
