# neo14 P1 `get_data` slowdown — ROOT CAUSE + FIX (2026-07-13) [RESOLVED]

## Symptom (confirmed, faithful cluster conditions)
neo14 P1 sim ~2× slower than the neo12 golden, entirely in `get_data`/serialization/gather (NEST compute
identical). test3 nt12 blank `get_data` ~158 s (neo14) vs golden ~76 s (`mozaik-opt.sif`, neo 0.12, job
14614503 = 16:55). S500 tracked ~1.5× slow.

## Root cause (cProfile of the real get_data — getdata_rank0.prof)
Hot chain: `sheet.get_data` → pyNN `population.get_data` → `recording.get` → **`gather_blocks`** (MPI gather
+ merge of per-rank neo Blocks) → **`neo.core.container.merge`** → per-spiketrain `SpikeTrainList.append`
→ `ObjectList._handle_append` → **`ObjectList._contains`**.

`neo/core/objectlist.py` `_contains` rebuilt `[id(item) for item in self._items]` and did `id(obj) in …`
on **every** append → O(N) per append → **O(N²)** over the ~37,500 spiketrains merged from 12 ranks
(profile: 946,386 appends; `id()` called 4.6 billion times; `_contains` 731 s of a 1115 s get_data).
neo 0.12 used plain lists (O(1) append, no dup-scan) → merge was O(N). Pure neo 0.13+ `ObjectList`
regression; it fires in the cross-rank MERGE, so it's invisible to single-rank construction micro-benchmarks
(which use the bulk `from_spike_time_array`) and dominant in the real 12-rank sim.

## Fix (mozaik/patches/neo-0.14.4/)
`_contains` → O(1) cached id-set tagged by `(id(self._items), len(self._items))`; kept in sync by
`_handle_append` (adds the incoming id, bumps len) so the append hot path is O(1); rebuilt O(N) only on
list-replacement / unexpected length change / unpickle; `__setitem__` (both classes) invalidates the cache.
Dup-detection preserved.

## Verification
- append micro-bench: O(N²)→O(1) — 117→962 µs/append (orig) vs flat ~2.3 µs (patched); 40k: 38.5 s → 0.09 s (~420×).
- test3 blank `get_data`: 158 s → **82/78/71 s = golden** (patched, inline).
- faithful sbatch test3 nt12 (patch bind-mounted, node-exclusive golden conditions): **913 s ≈ golden 1015 s**
  (per-presentation golden-identical), vs unpatched neo14 ~2× (~1900 s). Regression eliminated.
- **30/30 `tests/full_model` reference tests pass** (VogelsAbbott + LSV1M_tiny + 2024_LGN + stepcurrentmodule,
  spikes+voltages) with the patch. Correctness intact.

## Packaged
- `mozaik/patches/neo-0.14.4/` — patched files + diffs + README.
- `mozaik/mozaik-opt-neo14_2026-07_neopatch.def` → SIF `mozaik-sif/mozaik-opt-neopatch_2026-07-13.sif`
  (refmatch -O3 + patch baked via %post).

## Evidence artifacts (this dir)
- `26-07-13_neo_objectlist_bench.py` (append O(N²)→O(1)); `26-07-13_getdata_profile_top.txt` (cProfile top);
- `26-07-13_speed_patched.out` (blank 82/78/71 s); `26-07-13_patch_full_gate.out` (30/30);
- `26-07-13_patched-test3-*.{out,err}` (faithful 913 s); golden `../slurm_scale/mozaik-scale-nt12-14614503_12.err`.

## Prior (superseded) note
An interim read RULED OUT neo containers because the *initial* `_get_current_segment` uses the bulk
`from_spike_time_array` (not per-append). That was correct for construction but MISSED the **merge** path
(`gather_blocks`/`container.merge`), which does per-append and is where the O(N²) actually fires. The
cProfile corrected it. Lesson: profile the real path; don't infer from a micro-benchmark of one code path.
